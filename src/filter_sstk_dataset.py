'''
SSTK_Curation_Guide/curated_vs_rejected_logic.md
'''

import os
import glob
import json
import multiprocessing as mp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from tqdm import tqdm
from collections import Counter, defaultdict
from itertools import chain
import shutil

DEFAULT_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def _parse_gpu_ids_csv(csv_text):
    ids = []
    seen = set()
    for tok in str(csv_text or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            gid = int(tok)
        except ValueError:
            continue
        if gid < 0 or gid in seen:
            continue
        seen.add(gid)
        ids.append(gid)
    return ids


def _resolve_tag_embed_runtime(args):
    """
    Decide whether tag embedding should run on multi-GPU and resolve devices.
    Returns (use_multi_gpu, target_devices, single_device, cuda_count).
    """
    cuda_count = 0
    try:
        import torch
        cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        cuda_count = 0

    req_multi = int(getattr(args, "tag_embed_multi_gpu", -1))
    requested_device = str(getattr(args, "tag_embed_device", "auto")).strip().lower()

    if req_multi == 1:
        use_multi_gpu = True
    elif req_multi == 0:
        use_multi_gpu = False
    else:
        use_multi_gpu = (cuda_count > 1 and requested_device != "cpu")

    gpu_ids = _parse_gpu_ids_csv(getattr(args, "tag_embed_gpu_ids", ""))
    if not gpu_ids:
        gpu_ids = list(range(cuda_count))
    else:
        gpu_ids = [gid for gid in gpu_ids if gid < cuda_count]

    target_devices = [f"cuda:{gid}" for gid in gpu_ids]
    if use_multi_gpu and len(target_devices) < 2:
        print(
            f"[tag-embed] multi-gpu requested but available target devices are insufficient: "
            f"{target_devices} (cuda_count={cuda_count}). Fallback to single-device."
        )
        use_multi_gpu = False

    if requested_device == "auto":
        single_device = "cuda" if cuda_count > 0 else "cpu"
    else:
        single_device = requested_device

    return use_multi_gpu, target_devices, single_device, cuda_count


def _encode_unique_tags(model, unique_tags, args, use_multi_gpu, target_devices):
    batch_size = max(1, int(getattr(args, "tag_embed_batch_size", 128)))
    chunk_size = int(getattr(args, "tag_embed_chunk_size", 0))

    if use_multi_gpu and target_devices:
        if not hasattr(model, "start_multi_process_pool") or not hasattr(model, "encode_multi_process"):
            print("[tag-embed] sentence-transformers multi-process API unavailable. Fallback to single-device.")
        else:
            pool = None
            try:
                print(
                    "[tag-embed] Encoding dataset unique tags with multi-GPU "
                    f"(devices={','.join(target_devices)}, batch_size={batch_size}, "
                    f"chunk_size={chunk_size if chunk_size > 0 else 'auto'})"
                )
                pool = model.start_multi_process_pool(target_devices=target_devices)
                kwargs = {"pool": pool, "batch_size": batch_size}
                if chunk_size > 0:
                    kwargs["chunk_size"] = chunk_size
                tag_embs = model.encode_multi_process(unique_tags, **kwargs)
                return tag_embs
            except Exception as e:
                print(f"[tag-embed] multi-gpu encode failed, fallback to single-device. reason={e}")
            finally:
                if pool is not None:
                    try:
                        model.stop_multi_process_pool(pool)
                    except Exception:
                        pass

    print(f"[tag-embed] Encoding dataset unique tags on single device (batch_size={batch_size})...")
    return model.encode(unique_tags, show_progress_bar=True, batch_size=batch_size)


def _get_tags_list(t_val):
    # Handle numpy arrays / lists first to avoid ValueError from pd.isna(array).
    if isinstance(t_val, (list, np.ndarray)):
        return [str(x) for x in t_val if x is not None and str(x).strip()]
    # Now safe to call scalar pd.isna.
    try:
        if pd.isna(t_val) or not t_val:
            return []
    except (TypeError, ValueError):
        return []
    if isinstance(t_val, str):
        return [x for x in t_val.split('|') if x.strip()]
    return []


def _resolve_cpu_workers(requested, total_items, auto_cap=0):
    cpu = os.cpu_count() or 1
    req = int(requested)
    if auto_cap is None or int(auto_cap) <= 0:
        auto_cap = cpu
    else:
        auto_cap = min(cpu, int(auto_cap))
    if req < 0:
        workers = cpu
    elif req == 0:
        workers = min(cpu, auto_cap)
    else:
        workers = req
    if total_items > 0:
        workers = min(workers, total_items)
    return max(1, workers)


def _iter_list_chunks(values, chunk_size):
    if chunk_size <= 0:
        chunk_size = 1
    n = len(values)
    for i in range(0, n, chunk_size):
        yield values[i:i + chunk_size]


_CATEGORY_MAP_STATE = {}


def _init_category_map_worker(
    tag_cat_probs,
    categories,
    multi_tags,
    single_exclusive,
    single_general,
    conf_min,
    tag_counts,
):
    global _CATEGORY_MAP_STATE
    _CATEGORY_MAP_STATE = {
        "tag_cat_probs": tag_cat_probs,
        "categories": list(categories),
        "multi_tags": set(multi_tags),
        "single_exclusive": set(single_exclusive),
        "single_general": set(single_general),
        "conf_min": float(conf_min),
        "tag_counts": tag_counts,
    }


def _assign_single_category_and_rarest(t_raw):
    st = _CATEGORY_MAP_STATE
    tags = _get_tags_list(t_raw)
    if not tags:
        return "other_ambiguous", 1.0

    rarest_freq = float(min((st["tag_counts"].get(t, 1) for t in tags), default=1.0))
    categories = st["categories"]
    S_c = {c: 0.0 for c in categories}
    tag_cat_probs = st["tag_cat_probs"]

    # Evidence aggregation S(c|i) = sum_t p(c|t)
    for t in tags:
        probs = tag_cat_probs.get(t)
        if probs is None:
            continue
        for c in categories:
            S_c[c] += probs.get(c, 0.0)

    sorted_c = sorted(S_c.items(), key=lambda x: x[1], reverse=True)
    c_hat = sorted_c[0][0]
    score_1st = sorted_c[0][1]
    score_2nd = sorted_c[1][1] if len(sorted_c) > 1 else 0.0

    if (score_1st - score_2nd) < st["conf_min"]:
        return "other_ambiguous", rarest_freq

    if c_hat == "people":
        has_single_exclusive = any(t in st["single_exclusive"] for t in tags)
        has_multi = any(t in st["multi_tags"] for t in tags)
        has_single_general = any(t in st["single_general"] for t in tags)

        if has_single_exclusive:
            return "people_single", rarest_freq
        if has_multi:
            # e.g., 'team', 'couple', 'friends' overrides basic 'man', 'woman' mentions.
            return "people_multi", rarest_freq
        if has_single_general:
            return "people_single", rarest_freq
        return "other_ambiguous", rarest_freq

    return c_hat, rarest_freq


def _map_category_chunk_worker(tags_chunk):
    cats = []
    rares = []
    for t_raw in tags_chunk:
        c, r = _assign_single_category_and_rarest(t_raw)
        cats.append(c)
        rares.append(r)
    return cats, rares


def _map_categories_and_rarest_parallel(
    tags_values,
    tag_cat_probs,
    categories,
    multi_tags,
    single_exclusive,
    single_general,
    conf_min,
    tag_counts,
    workers,
    chunk_size,
):
    total = len(tags_values)
    if total == 0:
        return [], []

    workers = _resolve_cpu_workers(workers, total_items=total, auto_cap=0)
    chunk_size = max(1, int(chunk_size))
    total_chunks = (total + chunk_size - 1) // chunk_size

    init_args = (
        tag_cat_probs,
        categories,
        multi_tags,
        single_exclusive,
        single_general,
        conf_min,
        dict(tag_counts),
    )

    def _run_single():
        _init_category_map_worker(*init_args)
        mapped = []
        for tags_chunk in tqdm(
            _iter_list_chunks(tags_values, chunk_size),
            total=total_chunks,
            desc="Mapping categories",
        ):
            mapped.append(_map_category_chunk_worker(tags_chunk))
        return mapped

    # On Linux, fork keeps large read-only structures mostly copy-on-write.
    # Spawn/forkserver would require serializing huge dicts, so prefer single in that case.
    start_methods = set(mp.get_all_start_methods())
    if workers <= 1:
        chunk_results = _run_single()
    elif "fork" not in start_methods:
        print(
            "[category-map] 'fork' start method unavailable. "
            "Fallback to single-process mapping to avoid massive spawn serialization."
        )
        chunk_results = _run_single()
    else:
        try:
            print(f"[category-map] multiprocessing enabled: workers={workers} chunk_size={chunk_size}")
            ctx = mp.get_context("fork")
            with ctx.Pool(
                processes=workers,
                initializer=_init_category_map_worker,
                initargs=init_args,
            ) as pool:
                iter_chunks = _iter_list_chunks(tags_values, chunk_size)
                chunk_results = list(
                    tqdm(
                        pool.imap(_map_category_chunk_worker, iter_chunks, chunksize=1),
                        total=total_chunks,
                        desc="Mapping categories",
                    )
                )
        except Exception as e:
            print(f"[category-map] multiprocessing failed. fallback to single-process. reason={e}")
            chunk_results = _run_single()

    final_categories = []
    rarest_freqs = []
    for cats, rares in chunk_results:
        final_categories.extend(cats)
        rarest_freqs.extend(rares)
    return final_categories, rarest_freqs


def _extract_samples_from_single_tar(task):
    import tarfile

    tar_name, group, tar_dir, bucket, samples_dir = task
    tar_path = os.path.join(tar_dir, tar_name)
    if not os.path.exists(tar_path):
        tar_path = os.path.join(tar_dir, bucket, tar_name)

    targets = {str(image_id): str(pool_name) for image_id, pool_name in group}
    if not os.path.exists(tar_path):
        return {"tar_name": tar_name, "requested": len(targets), "extracted": 0, "missing": len(targets), "error": ""}

    extracted = 0
    found = set()
    err = ""
    try:
        # Use streaming read 'r|' which is tremendously faster for extraction than 'r'.
        with tarfile.open(tar_path, 'r|') as tf:
            for member in tf:
                if not member.isfile():
                    continue
                basename = os.path.basename(member.name)
                stem, ext = os.path.splitext(basename)
                if stem not in targets:
                    continue
                fobj = tf.extractfile(member)
                if fobj is None:
                    continue
                pool_name = targets[stem]
                out_path = os.path.join(samples_dir, pool_name, f"{stem}{ext.lower()}")
                with open(out_path, "wb") as wf:
                    wf.write(fobj.read())
                found.add(stem)
                extracted += 1
                if extracted >= len(targets):
                    break
    except Exception as e:
        err = str(e)

    missing = len(targets) - len(found)
    return {"tar_name": tar_name, "requested": len(targets), "extracted": extracted, "missing": missing, "error": err}

def process_file_pair(args):
    # Backward compatibility: old cache/jobs may pass 3-tuple.
    if len(args) == 3:
        sdp_file, train_file, tmp_dir = args
        require_train_match = True
    else:
        sdp_file, train_file, tmp_dir, require_train_match = args
    
    try:
        try:
            with open(sdp_file, 'r') as f:
                sdp_data = json.load(f).get('metadata', {})
        except Exception as e:
            print(f"Error parsing {sdp_file}: {e}")
            return False
            
        train_data = {}
        if not os.path.exists(train_file):
            if require_train_match:
                print(f"Warning: Train file not found {train_file}")
                return False
        else:
            try:
                with open(train_file, 'r') as f:
                    train_data = json.load(f).get('metadata', {})
            except Exception as e:
                print(f"Error parsing {train_file}: {e}")
                return False
            
        filtered_records = []
        fail_w = 0
        fail_aes = 0
        fail_dedup = 0
        
        for img_id, sdp_meta in sdp_data.items():
            # Quality Filters
            w = sdp_meta.get('width', 0) or 0
            h = sdp_meta.get('height', 0) or 0
            
            # Temporary relaxation/logging for debugging if all get filtered
            # if w < 512 or h < 512:
            #     continue
                
            aes_center = sdp_meta.get('aesthetic_score_center', 0) or 0
            aes_pad = sdp_meta.get('aesthetic_score_pad', 0) or 0
            
            if max(aes_center, aes_pad) < 5.0:
                continue
                
            if sdp_meta.get('image_dedup') is True:
                continue

            sstk_type = sdp_meta.get('sstk_type', '')
            if sstk_type != 'Photo':
                continue

            # Tags Extraction
            sstk_tags_str = sdp_meta.get('sstk_tags', '')
            sstk_tags = [t.strip() for t in sstk_tags_str.split(',') if t.strip()] if sstk_tags_str else []

            train_meta = train_data.get(img_id, {})
            merged_tags = train_meta.get('merged_tags', [])

            # Merge tags
            final_tags = list(set(sstk_tags + merged_tags))
            tar_name = os.path.basename(sdp_file).replace('.json', '.tar')

            filtered_records.append({
                'image_id': img_id,
                'width': w,
                'height': h,
                'aesthetic_score_center': aes_center,
                'aesthetic_score_pad': aes_pad,
                'sstk_type': sstk_type,
                'tags': "|".join(final_tags),
                'tar_name': tar_name
            })

        print(f"[{os.path.basename(sdp_file)}] Total: {len(sdp_data)}, Filtered out by res: {fail_w}, aes: {fail_aes}, dedup: {fail_dedup}. Passed: {len(filtered_records)}")

        if len(filtered_records) > 0:
            df_chunk = pd.DataFrame(filtered_records)
            chunk_path = os.path.join(tmp_dir, os.path.basename(sdp_file).replace('.json', '.parquet'))
            df_chunk.to_parquet(chunk_path, engine='pyarrow', index=False)

        return True
    except Exception as e:
        # Prevent silent crashes returning from worker
        import traceback
        print(f"Critical Worker Error processing {os.path.basename(sdp_file)}: {e}")
        traceback.print_exc()
        return False

def extract_and_save_samples(df_curated, df_rejected, args):
    import urllib.request

    out_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else "."
    samples_dir = os.path.join(out_dir, f"comparison_samples_{args.bucket}")
    os.makedirs(os.path.join(samples_dir, 'curated'), exist_ok=True)
    os.makedirs(os.path.join(samples_dir, 'rejected'), exist_ok=True)

    print("1) Sampling 20 images per category for qualitative comparison...")
    s_curated = df_curated.groupby('super_cat', group_keys=False).apply(lambda x: x.sample(n=min(len(x), 20), random_state=42))
    s_rejected = df_rejected.groupby('super_cat', group_keys=False).apply(lambda x: x.sample(n=min(len(x), 20), random_state=42))

    tasks = []
    for _, row in s_curated.iterrows():
        tasks.append((row['tar_name'], row['image_id'], 'curated', row))
    for _, row in s_rejected.iterrows():
        tasks.append((row['tar_name'], row['image_id'], 'rejected', row))

    tasks_df = pd.DataFrame([t[3] for t in tasks])
    tasks_df['pool'] = [t[2] for t in tasks]
    csv_meta_path = f"{samples_dir}/metadata_samples.csv"
    tasks_df.to_csv(csv_meta_path, index=False)
    print(f"2) Metadata for samples saved to: {csv_meta_path}")

    tasks_by_tar = {}
    for t in tasks:
        tar_name, image_id, pool_name, _row = t
        tasks_by_tar.setdefault(tar_name, []).append((str(image_id), str(pool_name)))

    extract_jobs = [
        (tar_name, group, args.tar_dir, args.bucket, samples_dir)
        for tar_name, group in tasks_by_tar.items()
    ]
    total_jobs = len(extract_jobs)
    extract_workers = _resolve_cpu_workers(
        requested=int(getattr(args, "sample_extract_workers", 0)),
        total_items=total_jobs,
        auto_cap=0,
    )

    print(
        f"3) Extracting sample images from {total_jobs} tars "
        f"(workers={extract_workers}). This may take a minute..."
    )
    extracted_total = 0
    missing_total = 0
    failed_jobs = 0

    if total_jobs > 0 and extract_workers > 1:
        with ThreadPoolExecutor(max_workers=extract_workers) as ex:
            futures = [ex.submit(_extract_samples_from_single_tar, job) for job in extract_jobs]
            for fut in tqdm(as_completed(futures), total=total_jobs, desc="Extracting sample images"):
                r = fut.result()
                extracted_total += int(r.get("extracted", 0))
                missing_total += int(r.get("missing", 0))
                if r.get("error"):
                    failed_jobs += 1
                    print(f"Failed to extract from tar {r.get('tar_name')}: {r.get('error')}")
    else:
        for job in tqdm(extract_jobs, total=total_jobs, desc="Extracting sample images"):
            r = _extract_samples_from_single_tar(job)
            extracted_total += int(r.get("extracted", 0))
            missing_total += int(r.get("missing", 0))
            if r.get("error"):
                failed_jobs += 1
                print(f"Failed to extract from tar {r.get('tar_name')}: {r.get('error')}")

    print(
        f"3a) Sample extraction summary: extracted={extracted_total} "
        f"missing={missing_total} failed_tars={failed_jobs}"
    )

    print("4) Generating HTML comparison report...")
    rel_samples_dir = os.path.basename(samples_dir)
    html_lines = [
        "<html><head><style>",
        "body { font-family: Arial, sans-serif; background-color: #f4f4f9; color: #333; margin: 20px; }",
        ".cat-section { margin-bottom: 50px; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }",
        ".pool-section { display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 20px; }",
        ".img-card { border: 1px solid #ddd; border-radius: 8px; padding: 10px; width: 250px; background: #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }",
        ".img-card img { max-width: 100%; height: 200px; object-fit: cover; border-radius: 4px; margin-bottom: 10px; }",
        ".img-card p { font-size: 0.9em; line-height: 1.4; margin: 0; word-break: break-all; }",
        "h1 { text-align: center; color: #2c3e50; }",
        "h2 { border-bottom: 2px solid #3498db; padding-bottom: 10px; color: #34495e; }",
        "h3 { color: #e74c3c; margin-top: 20px; }",
        "h3.curated-title { color: #27ae60; }",
        "</style></head><body><h1>Curated vs Rejected Comparison</h1>"
    ]
    all_cats = sorted(pd.concat([s_curated['super_cat'], s_rejected['super_cat']]).unique())

    for cat in all_cats:
        html_lines.append(f"<div class='cat-section'><h2>Category: {cat}</h2>")

        # Curated
        html_lines.append("<h3 class='curated-title'>Curated Pool (Accepted)</h3><div class='pool-section'>")
        cat_c = s_curated[s_curated['super_cat'] == cat]
        for _, row in cat_c.iterrows():
            img_path = f"curated/{row['image_id']}.jpg"
            aes = max(row['aesthetic_score_center'], row['aesthetic_score_pad'])
            raw_tags = row['tags']
            if isinstance(raw_tags, str):
                tags_list = raw_tags.split('|') if raw_tags else []
            else:
                tags_list = list(raw_tags) if raw_tags is not None else []
            tags = ", ".join(tags_list[:8]) + ("..." if len(tags_list) > 8 else "")
            sstk_type = row.get('sstk_type', 'N/A')
            html_lines.append(f"<div class='img-card'><img src='{rel_samples_dir}/{img_path}' loading='lazy'><p><b>ID:</b> {row['image_id']}<br><b>Type:</b> {sstk_type}<br><b>AES:</b> {aes:.2f}<br><b>DIMS:</b> {row['width']}x{row['height']}<br><b>TAGS:</b> {tags}</p></div>")
        html_lines.append("</div>")

        # Rejected
        html_lines.append("<h3>Rejected Pool (Discarded by Aesthetics/Long-Tail Cut)</h3><div class='pool-section'>")
        cat_r = s_rejected[s_rejected['super_cat'] == cat]
        for _, row in cat_r.iterrows():
            img_path = f"rejected/{row['image_id']}.jpg"
            aes = max(row['aesthetic_score_center'], row['aesthetic_score_pad'])
            raw_tags = row['tags']
            if isinstance(raw_tags, str):
                tags_list = raw_tags.split('|') if raw_tags else []
            else:
                tags_list = list(raw_tags) if raw_tags is not None else []
            tags = ", ".join(tags_list[:8]) + ("..." if len(tags_list) > 8 else "")
            sstk_type = row.get('sstk_type', 'N/A')
            html_lines.append(f"<div class='img-card'><img src='{rel_samples_dir}/{img_path}' loading='lazy'><p><b>ID:</b> {row['image_id']}<br><b>Type:</b> {sstk_type}<br><b>AES:</b> {aes:.2f}<br><b>DIMS:</b> {row['width']}x{row['height']}<br><b>TAGS:</b> {tags}</p></div>")
        html_lines.append("</div>")

        html_lines.append("</div>")

    html_lines.append("</body></html>")
    out_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else "."
    html_path = os.path.join(out_dir, f"comparison_report_{args.bucket}.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(html_lines))
    print(f"5) Qualitative HTML comparison report generated: {html_path}")


def export_curated_images(df_curated, args):
    image_dir = (args.save_curated_images_dir or "").strip()
    if not image_dir:
        return

    os.makedirs(image_dir, exist_ok=True)
    skip_existing = bool(int(args.save_curated_images_skip_existing))
    print(f"Exporting curated images to local dir: {image_dir}")

    if "tar_name" not in df_curated.columns:
        print("[export_curated_images] Missing tar_name column. Skipping.")
        return

    by_tar = defaultdict(set)
    for tar_name, image_id in zip(df_curated["tar_name"].astype(str), df_curated["image_id"].astype(str)):
        by_tar[tar_name].add(image_id)

    existing_stems = _collect_existing_image_stems(image_dir) if skip_existing else set()
    skipped = 0
    jobs = []
    for tar_name, id_set in by_tar.items():
        pending_ids = set(id_set)
        if existing_stems:
            already_saved = pending_ids & existing_stems
            if already_saved:
                skipped += len(already_saved)
                pending_ids -= already_saved
        if pending_ids:
            jobs.append((tar_name, sorted(pending_ids), args.tar_dir, args.bucket, image_dir))

    total_jobs = len(jobs)
    workers = _resolve_cpu_workers(
        requested=int(getattr(args, "save_curated_images_workers", 0)),
        total_items=total_jobs,
        auto_cap=int(getattr(args, "save_curated_images_auto_workers_cap", 8)),
    )
    chunksize = max(1, int(getattr(args, "save_curated_images_chunksize", 1)))

    print(
        f"[export_curated_images] jobs={total_jobs} workers={workers} "
        f"chunksize={chunksize} pre_skipped={skipped}"
    )

    saved = 0
    missing = 0
    failed_jobs = 0
    job_iter = None
    pool = None

    try:
        if total_jobs > 0 and workers > 1:
            pool = Pool(processes=workers)
            job_iter = pool.imap_unordered(_export_curated_images_from_single_tar, jobs, chunksize=chunksize)
        else:
            job_iter = map(_export_curated_images_from_single_tar, jobs)

        for result in tqdm(job_iter, total=total_jobs, desc="Export curated images"):
            saved += int(result.get("saved", 0))
            missing += int(result.get("missing", 0))
            if result.get("error"):
                failed_jobs += 1
                if failed_jobs <= 20:
                    print(
                        "[export_curated_images] Failed to export from "
                        f"{result.get('tar_name')}: {result.get('error')}"
                    )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    if failed_jobs > 20:
        print(f"[export_curated_images] ... and {failed_jobs - 20} more tar export errors")

    print(
        f"[export_curated_images] done. saved={saved} skipped={skipped} "
        f"missing={missing} total_targets={len(df_curated)} failed_tars={failed_jobs}"
    )


def _collect_existing_image_stems(image_dir):
    stems = set()
    try:
        with os.scandir(image_dir) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                stem, ext = os.path.splitext(entry.name)
                if stem and ext.lower() in DEFAULT_IMAGE_EXTS:
                    stems.add(stem)
    except FileNotFoundError:
        return set()
    return stems


def _resolve_curated_tar_path(tar_dir, bucket, tar_name):
    candidates = [
        os.path.join(tar_dir, tar_name),
        os.path.join(tar_dir, bucket, tar_name) if bucket else "",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return ""


def _export_curated_images_from_single_tar(job):
    import tarfile

    tar_name, image_ids, tar_dir, bucket, image_dir = job
    targets = set(image_ids)
    tar_path = _resolve_curated_tar_path(tar_dir, bucket, tar_name)
    if not tar_path:
        return {"saved": 0, "missing": len(targets), "tar_name": tar_name, "error": "tar_not_found"}

    saved = 0
    try:
        with tarfile.open(tar_path, "r|") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                base = os.path.basename(member.name)
                stem, ext = os.path.splitext(base)
                if stem not in targets:
                    continue
                ext = ext.lower()
                if ext not in DEFAULT_IMAGE_EXTS:
                    # SSTK tar member order is often: .desc/.id/.jpg/.tags.
                    # Only export real image payloads.
                    continue
                fobj = tf.extractfile(member)
                if fobj is None:
                    continue
                out_path = os.path.join(image_dir, f"{stem}{ext}")
                with open(out_path, "wb") as wf:
                    shutil.copyfileobj(fobj, wf, length=1024 * 1024)
                saved += 1
                targets.remove(stem)
                if not targets:
                    break
    except Exception as e:
        return {"saved": saved, "missing": len(targets), "tar_name": tar_name, "error": str(e)}

    return {"saved": saved, "missing": len(targets), "tar_name": tar_name, "error": ""}


def _dir_has_parquet(path):
    if not path or not os.path.isdir(path):
        return False
    try:
        for name in os.listdir(path):
            if name.lower().endswith(".parquet"):
                return True
    except Exception:
        return False
    return False


def _pick_existing_file(candidates):
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return ""


def _pick_existing_dir(candidates, require_parquet=False):
    for p in candidates:
        if not p or not os.path.isdir(p):
            continue
        if require_parquet and not _dir_has_parquet(p):
            continue
        return p
    return ""


def _resolve_filter_cache_paths(out_dir, bucket):
    """
    Canonical cache root:
      <out_dir>/cache/filter/
    Backward-compatible fallback:
      - <out_dir>/*
      - <out_dir>/Temp/*
      - <out_dir>/Temp/cleanup_*/*
    """
    cache_root = os.path.join(out_dir, "cache", "filter")
    os.makedirs(cache_root, exist_ok=True)

    canonical_tmp_dir = os.path.join(cache_root, f"tmp_parquets_{bucket}")
    canonical_df_cache = os.path.join(cache_root, f"df_mapped_cache_{bucket}.parquet")
    canonical_tag_cache = os.path.join(cache_root, f"tag_cat_probs_cache_{bucket}.pkl")

    cleanup_df = sorted(
        glob.glob(os.path.join(out_dir, "Temp", "cleanup_*", f"df_mapped_cache_{bucket}.parquet")),
        key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0,
        reverse=True,
    )
    cleanup_tag = sorted(
        glob.glob(os.path.join(out_dir, "Temp", "cleanup_*", f"tag_cat_probs_cache_{bucket}.pkl")),
        key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0,
        reverse=True,
    )
    cleanup_tmp = sorted(
        glob.glob(os.path.join(out_dir, "Temp", "cleanup_*", f"tmp_parquets_{bucket}")),
        key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0,
        reverse=True,
    )

    df_candidates = [
        canonical_df_cache,
        os.path.join(out_dir, f"df_mapped_cache_{bucket}.parquet"),
        os.path.join(out_dir, "Temp", f"df_mapped_cache_{bucket}.parquet"),
        *cleanup_df,
    ]
    tag_candidates = [
        canonical_tag_cache,
        os.path.join(out_dir, f"tag_cat_probs_cache_{bucket}.pkl"),
        os.path.join(out_dir, "Temp", f"tag_cat_probs_cache_{bucket}.pkl"),
        *cleanup_tag,
    ]
    tmp_candidates = [
        canonical_tmp_dir,
        os.path.join(out_dir, f"tmp_parquets_{bucket}"),
        os.path.join(out_dir, "Temp", f"tmp_parquets_{bucket}"),
        *cleanup_tmp,
    ]

    df_cache_read = _pick_existing_file(df_candidates)
    tag_cache_read = _pick_existing_file(tag_candidates)
    tmp_dir_read = _pick_existing_dir(tmp_candidates, require_parquet=True)

    return {
        "cache_root": cache_root,
        "canonical_tmp_dir": canonical_tmp_dir,
        "canonical_df_cache": canonical_df_cache,
        "canonical_tag_cache": canonical_tag_cache,
        "df_cache_read": df_cache_read,
        "tag_cache_read": tag_cache_read,
        "tmp_dir_read": tmp_dir_read,
    }


def _promote_cache_file(src_path, dst_path):
    if not src_path or not os.path.exists(src_path):
        return
    if os.path.abspath(src_path) == os.path.abspath(dst_path):
        return
    if os.path.exists(dst_path):
        return
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    try:
        shutil.copy2(src_path, dst_path)
        print(f"[cache] promoted file: {src_path} -> {dst_path}")
    except Exception as e:
        print(f"[cache] promote failed: {src_path} -> {dst_path} ({e})")


def _promote_cache_dir(src_dir, dst_dir):
    if not src_dir or not os.path.isdir(src_dir):
        return
    if os.path.abspath(src_dir) == os.path.abspath(dst_dir):
        return
    if os.path.isdir(dst_dir) and _dir_has_parquet(dst_dir):
        return
    os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
    try:
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        print(f"[cache] promoted dir: {src_dir} -> {dst_dir}")
    except Exception as e:
        print(f"[cache] promote dir failed: {src_dir} -> {dst_dir} ({e})")

def main():
    parser = argparse.ArgumentParser(description="Filter Shutterstock Dataset")
    parser.add_argument('--sdp_dir', type=str, default="/media/jyju25/T7_4TB_JY/Projects_26/Dataset/SSTK/sdp-sstk", help="Path to sdp metadata dir")
    parser.add_argument('--train_dir', type=str, default="/media/jyju25/T7_4TB_JY/Projects_26/Dataset/SSTK/SSTK_train_json/v1.0.1", help="Path to train metadata dir")
    parser.add_argument('--tar_dir', type=str, default="/media/jyju25/T7_4TB_JY/Projects_26/Dataset/SSTK/tars", help="Path to tar images dir")
    parser.add_argument('--bucket', type=str, default="sstk_100", help="Specific bucket to process")
    parser.add_argument('--output', type=str, default="filtered_sstk_100.parquet", help="Output parquet path")
    parser.add_argument('--curated_pool_size', type=int, default=1000000, help="Target size for the curated pool")
    parser.add_argument('--top_percentile', type=float, default=0.5, help="Top percentile to keep per category (e.g. 0.5 for top 50%%)")
    parser.add_argument('--server_mode', type=int, default=1, help="If 1, no limits are applied. If not 1, limits the processed tar count for local debugging.")
    parser.add_argument('--save_curated_images_dir', type=str, default="", help="Optional local image export dir for curated pool")
    parser.add_argument('--save_curated_images_skip_existing', type=int, default=1, help="1=skip existing files while exporting curated images")
    parser.add_argument(
        '--save_curated_images_workers',
        type=int,
        default=0,
        help="CPU workers for curated image export (-1=all cores, 0=auto(cap), >0=fixed).",
    )
    parser.add_argument(
        '--save_curated_images_auto_workers_cap',
        type=int,
        default=8,
        help="Auto-mode max workers cap for curated image export.",
    )
    parser.add_argument(
        '--save_curated_images_chunksize',
        type=int,
        default=1,
        help="Multiprocessing chunksize for curated image export jobs.",
    )
    parser.add_argument(
        '--require_train_match',
        type=int,
        default=1,
        help="If 1, only process SDP files that have matching TRAIN metadata JSON by filename.",
    )
    parser.add_argument(
        '--tag_embed_multi_gpu',
        type=int,
        default=-1,
        help="-1=auto(use multi-gpu when >=2 GPUs), 0=single-device, 1=force multi-gpu.",
    )
    parser.add_argument(
        '--tag_embed_gpu_ids',
        type=str,
        default="",
        help="CSV GPU ids for tag embedding multi-gpu (e.g., 0,1,2,3). Empty=all visible GPUs.",
    )
    parser.add_argument(
        '--tag_embed_batch_size',
        type=int,
        default=128,
        help="SentenceTransformer encode batch size for tag embeddings.",
    )
    parser.add_argument(
        '--tag_embed_chunk_size',
        type=int,
        default=0,
        help="encode_multi_process chunk size (0=auto).",
    )
    parser.add_argument(
        '--tag_embed_device',
        type=str,
        default="auto",
        help="single-device mode target device (auto|cuda|cuda:0|cpu).",
    )
    parser.add_argument(
        '--category_map_workers',
        type=int,
        default=0,
        help="CPU workers for category mapping (-1=all cores, 0=auto(all available cores), >0=fixed).",
    )
    parser.add_argument(
        '--category_map_chunk_size',
        type=int,
        default=4096,
        help="Category mapping chunk size per worker task.",
    )
    parser.add_argument(
        '--sample_extract_workers',
        type=int,
        default=0,
        help="Thread workers for sample tar extraction (-1=all cores, 0=auto(all available cores), >0=fixed).",
    )

    args = parser.parse_args()
    require_train_match = bool(int(args.require_train_match))

    sdp_files = glob.glob(os.path.join(args.sdp_dir, args.bucket, "*.json"))

    # User's constraint for local debugging: Only map to the available .tar limit
    if args.server_mode != 1:
        valid_sdp_files = []
        for f in sdp_files:
            basename = os.path.basename(f)
            if basename.startswith('SSTK_') and basename.endswith('.json'):
                try:
                    num = int(basename.replace('SSTK_', '').replace('.json', ''))
                    if num <= 11:
                        valid_sdp_files.append(f)
                except ValueError:
                    pass
        sdp_files = sorted(valid_sdp_files)
    else:
        sdp_files = sorted(sdp_files)

    out_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else "."
    cache_info = _resolve_filter_cache_paths(out_dir, args.bucket)

    tmp_pq_dir = cache_info["canonical_tmp_dir"]  # write target (canonical)
    tmp_pq_dir_read = cache_info["tmp_dir_read"]  # read source (fallback allowed)
    df_mapped_cache_path = cache_info["canonical_df_cache"]  # canonical
    df_mapped_cache_read = cache_info["df_cache_read"]       # fallback allowed
    cache_file = cache_info["canonical_tag_cache"]           # canonical
    cache_file_read = cache_info["tag_cache_read"]           # fallback allowed

    # Promote legacy caches once, then always use canonical paths.
    _promote_cache_file(df_mapped_cache_read, df_mapped_cache_path)
    _promote_cache_file(cache_file_read, cache_file)
    _promote_cache_dir(tmp_pq_dir_read, tmp_pq_dir)

    if os.path.exists(df_mapped_cache_path):
        df_mapped_cache_read = df_mapped_cache_path
    if os.path.exists(cache_file):
        cache_file_read = cache_file
    if _dir_has_parquet(tmp_pq_dir):
        tmp_pq_dir_read = tmp_pq_dir

    print(f"[cache] root={cache_info['cache_root']}")
    print(f"[cache] mapped_df={df_mapped_cache_path} (read={df_mapped_cache_read or '<none>'})")
    print(f"[cache] tag_probs={cache_file} (read={cache_file_read or '<none>'})")
    print(f"[cache] tmp_parquets={tmp_pq_dir} (read={tmp_pq_dir_read or '<none>'})")

    file_pairs = []
    missing_train_files = []
    for sdp_file in sdp_files:
        basename = os.path.basename(sdp_file)
        train_file = os.path.join(args.train_dir, args.bucket, basename)
        if require_train_match and not os.path.exists(train_file):
            missing_train_files.append(sdp_file)
            continue
        file_pairs.append((sdp_file, train_file, tmp_pq_dir, require_train_match))

    print(
        f"Found {len(file_pairs)} file pairs to process in bucket '{args.bucket}' "
        f"(require_train_match={int(require_train_match)})."
    )
    if missing_train_files:
        print(
            f"Skipped {len(missing_train_files)} SDP files without matching TRAIN json. "
            f"Examples: {', '.join(os.path.basename(p) for p in missing_train_files[:5])}"
        )
    if require_train_match and len(file_pairs) == 0 and len(sdp_files) > 0:
        raise RuntimeError(
            "No matched SDP/TRAIN file pairs were found. "
            "Check --train_dir/--bucket path consistency."
        )

    # Limit number of processes to avoid OOM or BrokenPipe on high-core servers.
    # Servers with high core counts can easily OOM if memory per worker is ~2GB.
    num_workers = min(6, os.cpu_count() or 1)

    mapping_already_done = False

    mapped_cache_for_read = ""
    if os.path.exists(df_mapped_cache_path):
        mapped_cache_for_read = df_mapped_cache_path
    elif df_mapped_cache_read and os.path.exists(df_mapped_cache_read):
        mapped_cache_for_read = df_mapped_cache_read

    if mapped_cache_for_read:
        print(f"\n[CACHE DETECTED] Found fully mapped cache {mapped_cache_for_read}.")
        print("Skipping multiprocessing JSON parsing AND category mapping phases completely.")
        df = pd.read_parquet(mapped_cache_for_read)
        print(f"Total valid mapped images loaded from cache: {len(df)}")
        mapping_already_done = True
    else:
        if os.path.exists(cache_file) and _dir_has_parquet(tmp_pq_dir_read):
            print(
                f"Found tag cache {cache_file} + temporary parquet dir {tmp_pq_dir_read}. "
                "Skipping multiprocessing JSON parsing phase."
            )
        else:
            os.makedirs(tmp_pq_dir, exist_ok=True)
            print(f"Initializing multiprocessing Pool with {num_workers} workers...")
            try:
                with Pool(processes=num_workers, maxtasksperchild=1) as pool:
                    # We don't accumulate anything in memory anymore, just iterate the bar
                    for result in tqdm(pool.imap_unordered(process_file_pair, file_pairs, chunksize=1), total=len(file_pairs)):
                        pass
            except Exception as e:
                import traceback
                print(f"Fatal exception during parallel file processing: {e}")
                traceback.print_exc()
                raise e
            tmp_pq_dir_read = tmp_pq_dir

        # Read back chunked parquets safely
        print(f"Loading temporary parquets from {tmp_pq_dir_read}...")
        import pyarrow.dataset as ds
        try:
            dataset = ds.dataset(tmp_pq_dir_read, format="parquet")
            df = dataset.to_table().to_pandas()
        except Exception as e:
            print(f"No valid Parquet chunks found or failed to load them: {e}")
            df = pd.DataFrame() # empty DataFrame

        print(f"Total valid images after initial filter: {len(df)}")

    if not mapping_already_done and len(df) > 0 and args.curated_pool_size > 0:
        print("Calculating aesthetic scores...")
        df['aes_score'] = df[['aesthetic_score_center', 'aesthetic_score_pad']].max(axis=1)

        print("Assigning super categories (MVP: Rule + Embedding + people_single/multi split)...")
        # 1. Define Categories and Seed Words (MVP rule-based labels)
        SEED_STRONG = {
            'people': ['person', 'people', 'man', 'woman', 'boy', 'girl', 'face', 'portrait', 'selfie', 'couple', 'family', 'crowd', 'team'],
            'animals': ['dog', 'cat', 'bird', 'wildlife', 'animal', 'pet', 'horse', 'fish', 'insect', 'reptile', 'cow', 'puppy', 'kitten'],
            'food': ['food', 'meal', 'dish', 'restaurant', 'fruit', 'vegetable', 'dessert', 'coffee', 'bread', 'meat'],
            'landscape_nature': ['mountain', 'forest', 'beach', 'ocean', 'river', 'sunset', 'sky', 'lake', 'nature', 'tree', 'sun', 'water', 'plant', 'flower', 'snow'],
            'architecture_exterior': ['building', 'architecture', 'skyscraper', 'bridge', 'landmark', 'facade', 'cityscape', 'exterior', 'city', 'town', 'street'],
            'indoor_interior': ['interior', 'living room', 'bedroom', 'kitchen', 'office', 'indoor', 'furniture', 'room'],
            'sports': ['sport', 'soccer', 'basketball', 'tennis', 'athlete', 'gym', 'running', 'baseball', 'fitness', 'workout', 'player', 'training'],
            'documents_text': ['document', 'paper', 'invoice', 'receipt', 'form', 'contract', 'text', 'typography', 'handwriting', 'banner', 'label', 'poster', 'writing', 'sign', 'letter', 'number'],
            'product_object': ['product', 'device', 'smartphone', 'laptop', 'gadget', 'bottle', 'cosmetics', 'clothing', 'shoes', 'object', 'item', 'tool'],
            'transportation': ['car', 'train', 'airplane', 'vehicle', 'boat', 'truck', 'bus', 'transportation']
        }

        CATEGORIES = list(SEED_STRONG.keys())

        # Unique tags
        print("Gathering unique tags...")
        tag_counts = Counter()
        for t_raw in df['tags']:
            t_list = _get_tags_list(t_raw)
            if t_list:
                tag_counts.update(t_list)

        unique_tags = list(tag_counts.keys())

        import pickle
        # cache_file path is already defined globally above

        cache_valid = False
        tag_cat_probs = {}
        if os.path.exists(cache_file):
            print(f"Loading cached tag probabilities from {cache_file}...")
            with open(cache_file, 'rb') as f:
                tag_cat_probs = pickle.load(f)
            if all(t in tag_cat_probs for t in unique_tags):
                cache_valid = True
            else:
                print("Cache is missing some tags. Recomputing...")

        if not cache_valid:
            print(f"Loading SentenceTransformer for {len(unique_tags)} unique tags...")
            from sentence_transformers import SentenceTransformer
            use_multi_gpu, target_devices, single_device, cuda_count = _resolve_tag_embed_runtime(args)
            model_device = "cpu" if use_multi_gpu else single_device
            print(
                "[tag-embed] runtime "
                f"cuda_count={cuda_count} "
                f"multi_gpu={int(use_multi_gpu)} "
                f"target_devices={target_devices if target_devices else [model_device]} "
                f"device_for_model={model_device}"
            )

            # Use a highly-efficient, lightweight embedding model for MVP.
            model = SentenceTransformer('all-MiniLM-L6-v2', device=model_device)

            print("Generating prototype embeddings...")
            cat_prototypes = {}
            for c, words in SEED_STRONG.items():
                emb = model.encode(words, show_progress_bar=False)
                cat_prototypes[c] = emb.mean(axis=0)
                cat_prototypes[c] /= np.linalg.norm(cat_prototypes[c]) + 1e-9

            print("Encoding dataset unique tags (this runs once per run)...")
            tag_embs = _encode_unique_tags(model, unique_tags, args, use_multi_gpu, target_devices)

            print("Pre-computing tag -> category probabilities...")
            gamma = 15.0 # softmax sharpness
            alpha = 0.8 # rule vs embedding mixture weight

            for i, t in enumerate(unique_tags):
                emb = tag_embs[i]
                emb /= (np.linalg.norm(emb) + 1e-9)

                # Embedding-based Probability
                cos_sims = {c: np.dot(emb, cat_prototypes[c]) for c in CATEGORIES}
                exp_sims = {c: np.exp(gamma * sim) for c, sim in cos_sims.items()}
                sum_exp = sum(exp_sims.values())
                p_emb = {c: exp_sims[c]/sum_exp for c in CATEGORIES}

                # Rule-based Probability
                p_rule = {c: 0.0 for c in CATEGORIES}
                matched_c = [c for c, words in SEED_STRONG.items() if t in words]
                if len(matched_c) > 0:
                    for c in matched_c:
                        p_rule[c] = 1.0 / len(matched_c)

                # Hybrid mix
                tag_cat_probs[t] = {c: alpha * p_rule[c] + (1 - alpha) * p_emb[c] for c in CATEGORIES}

            print(f"Saving tag probabilities to {cache_file} for future runs...")
            with open(cache_file, 'wb') as f:
                pickle.dump(tag_cat_probs, f)

        print("Assigning images to L1 Categories...")
        MULTI_TAGS = {'group', 'crowd', 'team', 'friends', 'meeting', 'audience', 'people', 'couple', 'twins', 'two', 'three', 'four', 'men', 'women', 'girls', 'boys', 'kids', 'children', 'adults', 'males', 'females', 'parents'}
        SINGLE_EXCLUSIVE = {'one person', 'alone', 'single', 'solo', 'selfie', 'only female', 'only male', 'one man', 'one woman'}
        SINGLE_GENERAL = {'portrait', 'headshot', 'person', 'man', 'woman', 'boy', 'girl', 'child', 'baby', 'gentleman', 'lady', 'guy', 'female', 'male', 'adult'}

        CONF_MIN = 0.1 # Ambiguous fallback threshold
        print(
            "[category-map] preparing CPU mapping "
            f"(workers={args.category_map_workers}, chunk_size={args.category_map_chunk_size})"
        )
        tags_values = df['tags'].tolist()
        final_categories, rarest_freqs = _map_categories_and_rarest_parallel(
            tags_values=tags_values,
            tag_cat_probs=tag_cat_probs,
            categories=CATEGORIES,
            multi_tags=MULTI_TAGS,
            single_exclusive=SINGLE_EXCLUSIVE,
            single_general=SINGLE_GENERAL,
            conf_min=CONF_MIN,
            tag_counts=tag_counts,
            workers=args.category_map_workers,
            chunk_size=args.category_map_chunk_size,
        )
        df['super_cat'] = final_categories
        df['rarest_freq'] = rarest_freqs

        print("Calculating long-tail oversampling weights...")
        df['sampling_weight'] = 1.0 / np.sqrt(df['rarest_freq'])
        
        print(f"Saving fully mapped DataFrame cache to {df_mapped_cache_path}...")
        df.to_parquet(df_mapped_cache_path, index=False)
        
    if len(df) > 0 and args.curated_pool_size > 0:
        df_initial = df.copy()
        print(f"Applying category-wise aesthetic percentile filtering (top {args.top_percentile*100}%)...")
        cat_thresholds = df.groupby('super_cat')['aes_score'].transform(lambda x: x.quantile(1.0 - args.top_percentile))
        df_filtered = df[df['aes_score'] >= cat_thresholds].copy()
        
        print(f"Images remaining after percentile cut: {len(df_filtered)}")
        
        if len(df_filtered) > args.curated_pool_size:
            print(f"Stratified sampling to reach {args.curated_pool_size} images...")
            cats = df_filtered['super_cat'].value_counts()
            
            target = args.curated_pool_size
            cap = 0
            n_cats = len(cats)
            cats_sorted = cats.sort_values(ascending=True)
            for i, count in enumerate(cats_sorted):
                alloc = target // (n_cats - i)
                if count <= alloc:
                    target -= count
                else:
                    cap = alloc
                    break
                    
            if cap == 0 and target > 0:
                cap = max(1, target // n_cats)
                
            def sample_group(g):
                n_samples = min(len(g), cap)
                weights = g['sampling_weight']
                return g.sample(n=n_samples, random_state=42, weights=weights)
                
            df_curated = df_filtered.groupby('super_cat', group_keys=False).apply(sample_group)
            
            rem = args.curated_pool_size - len(df_curated)
            if rem > 0:
                print(f"Sampling remaining {rem} images from leftover pool...")
                leftovers = df_filtered.loc[~df_filtered.index.isin(df_curated.index)]
                if len(leftovers) > 0:
                    df_curated = pd.concat([df_curated, leftovers.sample(n=min(rem, len(leftovers)), random_state=42, weights=leftovers['sampling_weight'])])
            
            df = df_curated
            print(f"Final pool size: {len(df)}")
            print("Category distribution:")
            print(df['super_cat'].value_counts())
        else:
            print("Filtered pool is already smaller than requested curated_pool_size, keeping all.")
            df = df_filtered
            
        print("Generating Curated vs Rejected Qualitative Comparison Extract...")
        try:
            # ONLY include images that actually failed the aesthetic 50% threshold cut. 
            # This avoids polluting the 'rejected' pool with top-50% images that were just dropped to meet curated_pool_size.
            df_rejected = df_initial[~df_initial['image_id'].isin(df_filtered['image_id'])].copy()
            extract_and_save_samples(df, df_rejected, args)
        except Exception as e:
            print(f"Failed to generate qualitative samples: {e}")
            raise e

        if str(args.save_curated_images_dir).strip():
            try:
                export_curated_images(df, args)
            except Exception as e:
                print(f"Failed to export curated images: {e}")
                raise e
            
        print("Generating summary CSV and visualization...")
        try:
            base_output = os.path.splitext(args.output)[0]
            summary_csv = f"{base_output}_summary.csv"
            summary_png = f"{base_output}_summary.png"
            
            summary_df = df.groupby('super_cat').agg(
                count=('image_id', 'count'),
                avg_aes_center=('aesthetic_score_center', 'mean'),
                avg_aes_pad=('aesthetic_score_pad', 'mean'),
                avg_width=('width', 'mean'),
                avg_height=('height', 'mean')
            ).reset_index()
            
            total_row = pd.DataFrame([{
                'super_cat': 'TOTAL',
                'count': len(df),
                'avg_aes_center': df['aesthetic_score_center'].mean(),
                'avg_aes_pad': df['aesthetic_score_pad'].mean(),
                'avg_width': df['width'].mean(),
                'avg_height': df['height'].mean()
            }])
            summary_df = pd.concat([summary_df, total_row], ignore_index=True)
            
            summary_df.to_csv(summary_csv, index=False)
            print(f"Summary CSV saved to {summary_csv}")
            
            plt.figure(figsize=(12, 6))
            cat_counts = df['super_cat'].value_counts()
            cat_counts.plot(kind='bar', color='skyblue', edgecolor='black')
            plt.title('Distribution of Super Categories in Curated Pool')
            plt.xlabel('Super Category')
            plt.ylabel('Number of Images')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(summary_png, dpi=300)
            plt.close()
            print(f"Summary visualization saved to {summary_png}")
        except Exception as e:
            print(f"Failed to generate summary/visualization: {e}")
    
    cols_to_drop = [c for c in ['aes_score', 'super_cat', 'rarest_freq', 'sampling_weight'] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    df.to_parquet(args.output, index=False)
    print(f"Saved results to {args.output}")

if __name__ == "__main__":
    main()
