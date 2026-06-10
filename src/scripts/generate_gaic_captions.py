#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic captions for GAIC images.")
    parser.add_argument("--image_dir", required=True, help="flat image dir (<image_id>.<ext>)")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--input_parquet", default="", help="optional parquet to preserve image ordering")
    parser.add_argument("--preset", default="local_efficient", choices=["local_efficient", "server_quality", "server_quality_florence"])
    parser.add_argument("--backend", default="", choices=["", "blip", "florence2"])
    parser.add_argument("--model_id", default="")
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|cuda:0")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--skip_existing", type=int, default=1)
    parser.add_argument("--progress", type=int, default=1)
    parser.add_argument(
        "--caption_prompt",
        default="",
        help="BLIP conditional prompt or Florence task token override. Empty uses preset default.",
    )
    parser.add_argument("--multi_gpu", type=int, default=0)
    parser.add_argument("--gpu_ids", default="")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument(
        "--existing_output_jsonl",
        default="",
        help="Optional existing output used only for skip-existing lookup when writing shard outputs.",
    )
    return parser.parse_args()


def resolve_device(value: str) -> str:
    text = str(value).strip().lower()
    if text in {"", "auto"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def resolve_dtype(value: str, device: str) -> torch.dtype:
    text = str(value).strip().lower()
    if text == "float16":
        return torch.float16
    if text == "bfloat16":
        return torch.bfloat16
    if text == "float32":
        return torch.float32
    if str(device).startswith("cuda"):
        return torch.float16
    return torch.float32


def clean_caption(text: str) -> str:
    out = " ".join(str(text or "").replace("\n", " ").split()).strip()
    if out.endswith("."):
        return out
    return out + "." if out else ""


@dataclass
class Preset:
    backend: str
    model_id: str
    caption_prompt: str


PRESETS: Dict[str, Preset] = {
    "local_efficient": Preset(
        backend="blip",
        model_id="Salesforce/blip-image-captioning-base",
        caption_prompt="",
    ),
    "server_quality": Preset(
        backend="blip",
        model_id="Salesforce/blip-image-captioning-large",
        caption_prompt="",
    ),
    "server_quality_florence": Preset(
        backend="florence2",
        model_id="microsoft/Florence-2-large-ft",
        caption_prompt="<DETAILED_CAPTION>",
    ),
}


def scan_images(image_dir: Path) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    seen: Dict[str, Path] = {}
    for path in sorted(image_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image_id = path.stem
        if image_id in seen:
            raise ValueError(f"duplicate image_id in image_dir: {image_id} -> {seen[image_id]} | {path}")
        seen[image_id] = path
        out.append((image_id, path))
    return out


def load_image_rows(image_dir: Path, input_parquet: Optional[Path], max_images: int) -> List[Tuple[str, Path]]:
    path_map = {image_id: path for image_id, path in scan_images(image_dir)}
    rows: List[Tuple[str, Path]] = []
    if input_parquet is not None and input_parquet.exists():
        df = pd.read_parquet(input_parquet)
        for image_id in df["image_id"].astype(str).tolist():
            path = path_map.get(image_id)
            if path is not None:
                rows.append((image_id, path))
    else:
        rows = list(path_map.items())
    if max_images > 0:
        rows = rows[:max_images]
    return rows


def load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            image_id = str(rec.get("image_id", "")).strip()
            if image_id:
                out.add(image_id)
    return out


def parse_gpu_ids(gpu_ids: str) -> List[str]:
    return [part.strip() for part in str(gpu_ids or "").split(",") if part.strip()]


def compute_shard_range(total: int, shard_index: int, num_shards: int) -> Tuple[int, int]:
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"invalid shard_index={shard_index} for num_shards={num_shards}")
    base = total // num_shards
    extra = total % num_shards
    start = shard_index * base + min(shard_index, extra)
    end = start + base + (1 if shard_index < extra else 0)
    return start, end


def resolve_worker_count(gpu_ids: Sequence[str], requested_workers: int) -> int:
    req = int(requested_workers)
    if req > 0:
        return max(1, req)
    if gpu_ids:
        return max(1, len(gpu_ids))
    return 1


def resolve_existing_output_path(output_jsonl: Path, existing_output_jsonl: Optional[Path]) -> Path:
    if existing_output_jsonl is not None:
        return existing_output_jsonl
    return output_jsonl


def load_rows_and_pending(
    *,
    image_dir: Path,
    input_parquet: Optional[Path],
    max_images: int,
    output_jsonl: Path,
    existing_output_jsonl: Optional[Path],
    skip_existing: int,
) -> Tuple[List[Tuple[str, Path]], set[str], List[Tuple[str, Path]]]:
    rows = load_image_rows(image_dir=image_dir, input_parquet=input_parquet, max_images=max_images)
    existing_ids = (
        load_existing_ids(resolve_existing_output_path(output_jsonl, existing_output_jsonl))
        if int(skip_existing) == 1
        else set()
    )
    pending = [(image_id, path) for image_id, path in rows if image_id not in existing_ids]
    return rows, existing_ids, pending


class BaseCaptioner:
    def __init__(
        self,
        *,
        model_id: str,
        device: str,
        dtype: torch.dtype,
        caption_prompt: str,
        max_new_tokens: int,
        num_beams: int,
    ):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.caption_prompt = caption_prompt
        self.max_new_tokens = int(max_new_tokens)
        self.num_beams = int(num_beams)

    def generate_batch(self, images: Sequence[Image.Image]) -> List[str]:
        raise NotImplementedError


class BlipCaptioner(BaseCaptioner):
    def __init__(
        self,
        *,
        model_id: str,
        device: str,
        dtype: torch.dtype,
        caption_prompt: str,
        max_new_tokens: int,
        num_beams: int,
    ):
        super().__init__(
            model_id=model_id,
            device=device,
            dtype=dtype,
            caption_prompt=caption_prompt,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        from transformers import BlipForConditionalGeneration, BlipProcessor

        self.processor = BlipProcessor.from_pretrained(model_id)
        model_path = Path(model_id) if str(model_id).strip() else Path("")
        has_safetensors = bool(model_path.is_dir() and any(model_path.glob("*.safetensors")))
        load_kwargs = {"torch_dtype": dtype}
        if has_safetensors:
            load_kwargs["use_safetensors"] = True
        self.model = BlipForConditionalGeneration.from_pretrained(model_id, **load_kwargs).to(device)
        self.model.eval()

    @torch.no_grad()
    def generate_batch(self, images: Sequence[Image.Image]) -> List[str]:
        prompt = self.caption_prompt.strip()
        if prompt:
            inputs = self.processor(images=list(images), text=[prompt] * len(images), return_tensors="pt", padding=True)
        else:
            inputs = self.processor(images=list(images), return_tensors="pt", padding=True)
        prepared = {}
        for key, value in inputs.items():
            if value.dtype.is_floating_point:
                prepared[key] = value.to(self.device, self.dtype)
            else:
                prepared[key] = value.to(self.device)
        output_ids = self.model.generate(
            **prepared,
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_beams,
            do_sample=False,
        )
        return [clean_caption(text) for text in self.processor.batch_decode(output_ids, skip_special_tokens=True)]


class Florence2Captioner(BaseCaptioner):
    def __init__(
        self,
        *,
        model_id: str,
        device: str,
        dtype: torch.dtype,
        caption_prompt: str,
        max_new_tokens: int,
        num_beams: int,
    ):
        super().__init__(
            model_id=model_id,
            device=device,
            dtype=dtype,
            caption_prompt=caption_prompt,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.task_prompt = caption_prompt.strip() or "<DETAILED_CAPTION>"
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def generate_batch(self, images: Sequence[Image.Image]) -> List[str]:
        inputs = self.processor(
            text=[self.task_prompt] * len(images),
            images=list(images),
            return_tensors="pt",
            padding=True,
        )
        prepared = {}
        for key, value in inputs.items():
            if value.dtype.is_floating_point:
                prepared[key] = value.to(self.device, self.dtype)
            else:
                prepared[key] = value.to(self.device)
        output_ids = self.model.generate(
            **prepared,
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_beams,
            do_sample=False,
        )
        generated = self.processor.batch_decode(output_ids, skip_special_tokens=False)
        captions: List[str] = []
        for text, image in zip(generated, images):
            parsed = self.processor.post_process_generation(
                text,
                task=self.task_prompt,
                image_size=(image.width, image.height),
            )
            if isinstance(parsed, dict):
                value = parsed.get(self.task_prompt, "")
                if isinstance(value, str):
                    captions.append(clean_caption(value))
                else:
                    captions.append(clean_caption(str(value)))
            else:
                captions.append(clean_caption(str(parsed)))
        return captions


def build_captioner(args: argparse.Namespace) -> BaseCaptioner:
    preset = PRESETS[str(args.preset)]
    backend = str(args.backend or preset.backend)
    model_id = str(args.model_id or preset.model_id)
    caption_prompt = str(args.caption_prompt if args.caption_prompt != "" else preset.caption_prompt)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    if backend == "blip":
        return BlipCaptioner(
            model_id=model_id,
            device=device,
            dtype=dtype,
            caption_prompt=caption_prompt,
            max_new_tokens=int(args.max_new_tokens),
            num_beams=int(args.num_beams),
        )
    if backend == "florence2":
        return Florence2Captioner(
            model_id=model_id,
            device=device,
            dtype=dtype,
            caption_prompt=caption_prompt,
            max_new_tokens=int(args.max_new_tokens),
            num_beams=int(args.num_beams),
        )
    raise ValueError(f"unsupported backend: {backend}")


def preview_caption_config(args: argparse.Namespace) -> Dict[str, str]:
    preset = PRESETS[str(args.preset)]
    backend = str(args.backend or preset.backend)
    model_id = str(args.model_id or preset.model_id)
    caption_prompt = str(args.caption_prompt if args.caption_prompt != "" else preset.caption_prompt)
    device = resolve_device(args.device)
    dtype = str(resolve_dtype(args.dtype, device)).replace("torch.", "")
    return {
        "backend": backend,
        "model_id": model_id,
        "caption_prompt": caption_prompt,
        "device": device,
        "dtype": dtype,
    }


def iter_batches(items: Sequence[Tuple[str, Path]], batch_size: int) -> Iterable[Sequence[Tuple[str, Path]]]:
    for start in range(0, len(items), max(1, batch_size)):
        yield items[start : start + max(1, batch_size)]


def process_one_shard(args: argparse.Namespace) -> Dict[str, object]:
    image_dir = Path(args.image_dir).resolve()
    output_jsonl = Path(args.output_jsonl).resolve()
    summary_json = Path(args.summary_json).resolve()
    input_parquet = Path(args.input_parquet).resolve() if str(args.input_parquet).strip() else None
    existing_output_jsonl = (
        Path(args.existing_output_jsonl).resolve() if str(args.existing_output_jsonl).strip() else None
    )

    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir not found: {image_dir}")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    rows, existing_ids, pending = load_rows_and_pending(
        image_dir=image_dir,
        input_parquet=input_parquet,
        max_images=int(args.max_images),
        output_jsonl=output_jsonl,
        existing_output_jsonl=existing_output_jsonl,
        skip_existing=int(args.skip_existing),
    )
    shard_start, shard_end = compute_shard_range(len(pending), int(args.shard_index), int(args.num_shards))
    shard_rows = pending[shard_start:shard_end]
    preview = preview_caption_config(args)

    if not shard_rows:
        summary = {
            "status": "ok",
            "image_dir": str(image_dir),
            "input_parquet": (str(input_parquet) if input_parquet is not None else ""),
            "output_jsonl": str(output_jsonl),
            "preset": str(args.preset),
            "backend": preview["backend"],
            "model_id": preview["model_id"],
            "caption_prompt": preview["caption_prompt"],
            "device": preview["device"],
            "dtype": preview["dtype"],
            "num_rows_total": len(rows),
            "num_existing_skipped": len(existing_ids.intersection({image_id for image_id, _ in rows})),
            "num_rows_pending_after_skip": len(pending),
            "num_rows_assigned_to_shard": 0,
            "num_rows_generated_this_run": 0,
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
        }
        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    captioner = build_captioner(args)
    append_mode = (
        int(args.skip_existing) == 1
        and resolve_existing_output_path(output_jsonl, existing_output_jsonl) == output_jsonl
    )
    if not append_mode and output_jsonl.exists():
        output_jsonl.unlink()

    written = 0
    total_batches = (len(shard_rows) + max(1, int(args.batch_size)) - 1) // max(1, int(args.batch_size))
    open_mode = "a" if append_mode else "w"
    with output_jsonl.open(open_mode, encoding="utf-8") as handle:
        iterator = iter_batches(shard_rows, int(args.batch_size))
        for batch in tqdm(
            iterator,
            total=total_batches,
            desc=f"gaic-caption[{args.shard_index}/{args.num_shards}]",
            disable=(int(args.progress) == 0 or not shard_rows),
        ):
            images = [Image.open(path).convert("RGB") for _, path in batch]
            try:
                captions = captioner.generate_batch(images)
            finally:
                for image in images:
                    image.close()
            for (image_id, path), caption in zip(batch, captions):
                rec = {
                    "image_id": image_id,
                    "caption": clean_caption(caption),
                    "caption_backend": captioner.__class__.__name__.replace("Captioner", "").lower(),
                    "caption_model_id": captioner.model_id,
                    "caption_prompt": captioner.caption_prompt,
                    "source_image_path": str(path),
                }
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

    summary = {
        "status": "ok",
        "image_dir": str(image_dir),
        "input_parquet": (str(input_parquet) if input_parquet is not None else ""),
        "output_jsonl": str(output_jsonl),
        "preset": str(args.preset),
        "backend": captioner.__class__.__name__.replace("Captioner", "").lower(),
        "model_id": captioner.model_id,
        "caption_prompt": captioner.caption_prompt,
        "device": captioner.device,
        "dtype": str(captioner.dtype).replace("torch.", ""),
        "num_rows_total": len(rows),
        "num_existing_skipped": len(existing_ids.intersection({image_id for image_id, _ in rows})),
        "num_rows_pending_after_skip": len(pending),
        "num_rows_assigned_to_shard": len(shard_rows),
        "num_rows_generated_this_run": written,
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def launch_multi_gpu(args: argparse.Namespace) -> Dict[str, object]:
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    if not gpu_ids:
        raise ValueError("--multi_gpu 1 requires --gpu_ids")
    workers = resolve_worker_count(gpu_ids, int(args.num_workers))
    if workers <= 1:
        single_args = argparse.Namespace(**vars(args))
        single_args.multi_gpu = 0
        single_args.num_shards = 1
        single_args.shard_index = 0
        return process_one_shard(single_args)

    output_jsonl = Path(args.output_jsonl).resolve()
    summary_json = Path(args.summary_json).resolve()
    shard_dir = output_jsonl.parent / f"{output_jsonl.name}.shards.{os.getpid()}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    print(f"[multi] gaic-caption shard mode enabled: gpu_ids={','.join(gpu_ids)} workers={workers}")

    procs: List[Tuple[subprocess.Popen[str], Path]] = []
    shard_outputs: List[Path] = []
    shard_summaries: List[Path] = []

    for i in range(workers):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        shard_out = shard_dir / f"gaic_caption_shard_{i}.jsonl"
        shard_sum = shard_dir / f"gaic_caption_summary_shard_{i}.json"
        shard_log = shard_dir / f"gaic_caption_shard_{i}.log"
        shard_outputs.append(shard_out)
        shard_summaries.append(shard_sum)

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--image_dir",
            str(Path(args.image_dir)),
            "--output_jsonl",
            str(shard_out),
            "--summary_json",
            str(shard_sum),
            "--input_parquet",
            str(args.input_parquet),
            "--preset",
            str(args.preset),
            "--backend",
            str(args.backend),
            "--model_id",
            str(args.model_id),
            "--device",
            "cuda:0",
            "--dtype",
            str(args.dtype),
            "--batch_size",
            str(int(args.batch_size)),
            "--max_new_tokens",
            str(int(args.max_new_tokens)),
            "--num_beams",
            str(int(args.num_beams)),
            "--max_images",
            str(int(args.max_images)),
            "--skip_existing",
            str(int(args.skip_existing)),
            "--progress",
            "1" if i == 0 and int(args.progress) != 0 else "0",
            "--caption_prompt",
            str(args.caption_prompt),
            "--multi_gpu",
            "0",
            "--num_shards",
            str(workers),
            "--shard_index",
            str(i),
            "--existing_output_jsonl",
            str(output_jsonl),
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        print(f"[multi] launch shard={i}/{workers} gpu={gpu_id} -> {shard_out}")
        with shard_log.open("w", encoding="utf-8") as log_handle:
            proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, text=True, env=env)
        procs.append((proc, shard_log))

    failed = False
    for proc, shard_log in procs:
        if proc.wait() != 0:
            failed = True
            print(f"[error] gaic-caption shard failed: log={shard_log}", file=sys.stderr)
    if failed:
        for _, shard_log in procs:
            if shard_log.exists():
                print(f"----- tail: {shard_log} -----", file=sys.stderr)
                try:
                    for line in shard_log.read_text(encoding="utf-8").splitlines()[-80:]:
                        print(line, file=sys.stderr)
                except Exception:
                    pass
        raise RuntimeError(f"one or more gaic-caption shards failed. logs under: {shard_dir}")

    merged_tmp = shard_dir / "gaic_caption_merged.jsonl"
    with merged_tmp.open("w", encoding="utf-8") as fout:
        if int(args.skip_existing) == 1 and output_jsonl.exists():
            with output_jsonl.open("r", encoding="utf-8") as existing_handle:
                for line in existing_handle:
                    fout.write(line)
        for shard_out in shard_outputs:
            if shard_out.exists():
                with shard_out.open("r", encoding="utf-8") as shard_handle:
                    for line in shard_handle:
                        fout.write(line)
    merged_tmp.replace(output_jsonl)

    shard_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in shard_summaries if path.exists()]
    base = shard_payloads[0] if shard_payloads else {}
    summary = dict(base)
    summary["output_jsonl"] = str(output_jsonl)
    summary["summary_json"] = str(summary_json)
    summary["num_rows_generated_this_run"] = int(
        sum(int(payload.get("num_rows_generated_this_run", 0)) for payload in shard_payloads)
    )
    summary["num_shards"] = workers
    summary["shard_index"] = -1
    summary["multi_gpu"] = {"enabled": True, "gpu_ids": list(gpu_ids), "workers": workers, "shard_dir": str(shard_dir)}
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    if int(args.multi_gpu) != 0:
        summary = launch_multi_gpu(args)
    else:
        summary = process_one_shard(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
