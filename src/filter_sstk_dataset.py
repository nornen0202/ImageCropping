import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import argparse
from tqdm import tqdm
from collections import Counter
from itertools import chain

def process_file_pair(args):
    sdp_file, train_file = args
    
    try:
        with open(sdp_file, 'r') as f:
            sdp_data = json.load(f).get('metadata', {})
    except Exception as e:
        print(f"Error parsing {sdp_file}: {e}")
        return []
        
    if not os.path.exists(train_file):
        return []

    try:
        with open(train_file, 'r') as f:
            train_data = json.load(f).get('metadata', {})
    except Exception as e:
        print(f"Error parsing {train_file}: {e}")
        return []
        
    filtered_records = []
    
    for img_id, sdp_meta in sdp_data.items():
        # Quality Filters
        w = sdp_meta.get('width', 0) or 0
        h = sdp_meta.get('height', 0) or 0
        
        if w < 512 or h < 512:
            continue
            
        aes_center = sdp_meta.get('aesthetic_score_center', 0) or 0
        aes_pad = sdp_meta.get('aesthetic_score_pad', 0) or 0
        
        if max(aes_center, aes_pad) < 5.0:
            continue
            
        if sdp_meta.get('image_dedup', True) == True:
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
            'tags': final_tags,
            'tar_name': tar_name
        })
        
    return filtered_records

def extract_and_save_samples(df_curated, df_rejected, args):
    import tarfile
    import urllib.request
    
    samples_dir = f"comparison_samples_{args.bucket}"
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
        tasks_by_tar.setdefault(t[0], []).append(t)
        
    print(f"3) Extracting sample images from {len(tasks_by_tar)} tars. This may take a minute...")
    for tar_name, group in tqdm(tasks_by_tar.items(), desc="Extracting sample images"):
        tar_path = os.path.join(args.tar_dir, args.bucket, tar_name)
        if not os.path.exists(tar_path):
            continue
        try:
            with tarfile.open(tar_path, 'r') as tf:
                for task in group:
                    tar_name, image_id, pool_name, row_meta = task
                    jpg_name = f"{image_id}.jpg" # Some tars might prepend keys, but usually basic match works:
                    try:
                        # Attempt to find member ending with image_id.jpg
                        matched_member = next((m for m in tf.getmembers() if m.name.endswith(jpg_name)), None)
                        if matched_member:
                            matched_member.name = jpg_name
                            tf.extract(matched_member, path=os.path.join(samples_dir, pool_name))
                    except Exception:
                        pass
        except Exception as e:
            print(f"Failed to read tar {tar_name}: {e}")
            
    print("4) Generating HTML comparison report...")
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
            tags_list = list(row['tags'])
            tags = ", ".join(tags_list[:8]) + ("..." if len(tags_list) > 8 else "")
            html_lines.append(f"<div class='img-card'><img src='{samples_dir}/{img_path}' loading='lazy'><p><b>ID:</b> {row['image_id']}<br><b>AES:</b> {aes:.2f}<br><b>DIMS:</b> {row['width']}x{row['height']}<br><b>TAGS:</b> {tags}</p></div>")
        html_lines.append("</div>")
        
        # Rejected
        html_lines.append("<h3>Rejected Pool (Discarded by Aesthetics/Long-Tail Cut)</h3><div class='pool-section'>")
        cat_r = s_rejected[s_rejected['super_cat'] == cat]
        for _, row in cat_r.iterrows():
            img_path = f"rejected/{row['image_id']}.jpg"
            aes = max(row['aesthetic_score_center'], row['aesthetic_score_pad'])
            tags_list = list(row['tags'])
            tags = ", ".join(tags_list[:8]) + ("..." if len(tags_list) > 8 else "")
            html_lines.append(f"<div class='img-card'><img src='{samples_dir}/{img_path}' loading='lazy'><p><b>ID:</b> {row['image_id']}<br><b>AES:</b> {aes:.2f}<br><b>DIMS:</b> {row['width']}x{row['height']}<br><b>TAGS:</b> {tags}</p></div>")
        html_lines.append("</div>")
        
        html_lines.append("</div>")
        
    html_lines.append("</body></html>")
    html_path = f"comparison_report_{args.bucket}.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(html_lines))
    print(f"5) Qualitative HTML comparison report generated: {html_path}")

def main():
    parser = argparse.ArgumentParser(description="Filter Shutterstock Dataset")
    parser.add_argument('--sdp_dir', type=str, default="/media/jyju25/T7_4TB_JY/Projects_26/Dataset/SSTK/sdp-sstk", help="Path to sdp metadata dir")
    parser.add_argument('--train_dir', type=str, default="/media/jyju25/T7_4TB_JY/Projects_26/Dataset/SSTK/SSTK_train_json/v1.0.1", help="Path to train metadata dir")
    parser.add_argument('--tar_dir', type=str, default="/media/jyju25/T7_4TB_JY/Projects_26/Dataset/SSTK/tars", help="Path to tar images dir")
    parser.add_argument('--bucket', type=str, default="sstk_100", help="Specific bucket to process")
    parser.add_argument('--output', type=str, default="filtered_sstk_100.parquet", help="Output parquet path")
    parser.add_argument('--curated_pool_size', type=int, default=1000000, help="Target size for the curated pool")
    parser.add_argument('--top_percentile', type=float, default=0.5, help="Top percentile to keep per category (e.g. 0.5 for top 50%)")
    
    args = parser.parse_args()
    
    sdp_files = glob.glob(os.path.join(args.sdp_dir, args.bucket, "*.json"))
    
    file_pairs = []
    for sdp_file in sdp_files:
        basename = os.path.basename(sdp_file)
        train_file = os.path.join(args.train_dir, args.bucket, basename)
        file_pairs.append((sdp_file, train_file))
        
    print(f"Found {len(file_pairs)} file pairs to process in bucket '{args.bucket}'.")
    
    all_records = []
    with Pool(processes=os.cpu_count()) as pool:
        for records in tqdm(pool.imap_unordered(process_file_pair, file_pairs), total=len(file_pairs)):
            all_records.extend(records)
            
    df = pd.DataFrame(all_records)
    print(f"Total valid images after initial filter: {len(df)}")
    
    if len(df) > 0 and args.curated_pool_size > 0:
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
        tag_counts = Counter(chain.from_iterable(df['tags']))
        unique_tags = list(tag_counts.keys())
        
        import pickle
        cache_file = f"tag_cat_probs_cache_{args.bucket}.pkl"
        
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
            # Use a highly-efficient, lightweight embedding model for MVP
            model = SentenceTransformer('all-MiniLM-L6-v2') 
            
            print("Generating prototype embeddings...")
            cat_prototypes = {}
            for c, words in SEED_STRONG.items():
                emb = model.encode(words, show_progress_bar=False)
                cat_prototypes[c] = emb.mean(axis=0)
                cat_prototypes[c] /= np.linalg.norm(cat_prototypes[c]) + 1e-9
                
            print("Encoding dataset unique tags (this runs once per run)...")
            tag_embs = model.encode(unique_tags, show_progress_bar=True, batch_size=2048)
            
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
        MULTI_TAGS = {'group', 'crowd', 'team', 'family', 'friends', 'meeting', 'audience', 'people'}
        SINGLE_TAGS = {'portrait', 'selfie', 'headshot', 'model', 'face', 'one person', 'person', 'man', 'woman', 'boy', 'girl'}
        
        CONF_MIN = 0.1 # Ambiguous fallback threshold
        final_categories = []
        
        # Process every image
        for tags in tqdm(df['tags'], total=len(df), desc="Mapping categories"):
            if not tags:
                final_categories.append('other_ambiguous')
                continue
                
            # Evidence aggregation S(c|i) = sum_t p(c|t)
            S_c = {c: 0.0 for c in CATEGORIES}
            for t in tags:
                probs = tag_cat_probs[t]
                for c in CATEGORIES:
                    S_c[c] += probs[c]
            
            sorted_c = sorted(S_c.items(), key=lambda x: x[1], reverse=True)
            c_hat = sorted_c[0][0]
            score_1st = sorted_c[0][1]
            score_2nd = sorted_c[1][1] if len(sorted_c) > 1 else 0
            
            # 1) Category Confidence Fallback
            if (score_1st - score_2nd) < CONF_MIN:
                final_categories.append('other_ambiguous')
                continue
                
            # 2) People Multi vs Single Heuristic Split
            if c_hat == 'people':
                s_multi = sum(1 for t in tags if t in MULTI_TAGS)
                s_single = sum(1 for t in tags if t in SINGLE_TAGS)
                if s_multi - s_single > 0:
                    final_categories.append('people_multi')
                elif s_single - s_multi > 0:
                    final_categories.append('people_single')
                else:
                    final_categories.append('other_ambiguous') # Fallback if tie or 0
            else:
                final_categories.append(c_hat)
                
        df['super_cat'] = final_categories

        print("Calculating rare tag frequencies for long-tail oversampling...")
        def get_rarest_tag_freq(tags):
            if not tags: return 1.0
            return float(min((tag_counts.get(t, 1) for t in tags), default=1.0))
            
        df['rarest_freq'] = [get_rarest_tag_freq(t) for t in df['tags']]
        df['sampling_weight'] = 1.0 / np.sqrt(df['rarest_freq'])
        
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
            df_rejected = df_initial[~df_initial['image_id'].isin(df['image_id'])].copy()
            extract_and_save_samples(df, df_rejected, args)
        except Exception as e:
            print(f"Failed to generate qualitative samples: {e}")
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
