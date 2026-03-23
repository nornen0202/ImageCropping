#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    parser.add_argument(
        "--caption_prompt",
        default="",
        help="BLIP conditional prompt or Florence task token override. Empty uses preset default.",
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
        self.model = BlipForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype).to(device)
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


def iter_batches(items: Sequence[Tuple[str, Path]], batch_size: int) -> Iterable[Sequence[Tuple[str, Path]]]:
    for start in range(0, len(items), max(1, batch_size)):
        yield items[start : start + max(1, batch_size)]


def main() -> None:
    args = parse_args()

    image_dir = Path(args.image_dir).resolve()
    output_jsonl = Path(args.output_jsonl).resolve()
    summary_json = Path(args.summary_json).resolve()
    input_parquet = Path(args.input_parquet).resolve() if str(args.input_parquet).strip() else None

    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir not found: {image_dir}")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    rows = load_image_rows(image_dir=image_dir, input_parquet=input_parquet, max_images=int(args.max_images))
    if int(args.skip_existing) != 1 and output_jsonl.exists():
        output_jsonl.unlink()
    existing_ids = load_existing_ids(output_jsonl) if int(args.skip_existing) == 1 else set()
    pending = [(image_id, path) for image_id, path in rows if image_id not in existing_ids]

    captioner = build_captioner(args)

    written = 0
    total_batches = (len(pending) + max(1, int(args.batch_size)) - 1) // max(1, int(args.batch_size))
    with output_jsonl.open("a", encoding="utf-8") as handle:
        for batch in tqdm(iter_batches(pending, int(args.batch_size)), total=total_batches, desc="gaic-caption", disable=not pending):
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
        "num_rows_generated_this_run": written,
        "num_rows_pending_after_skip": len(pending),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
