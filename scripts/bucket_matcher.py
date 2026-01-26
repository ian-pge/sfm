import h5py
import torch
import numpy as np
import sys
from pathlib import Path

def generate_bucketed_retrieval_pairs(global_features_path, output_pairs_path, num_matched=20):
    print(f"Generating bucketed retrieval pairs from {global_features_path}...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    buckets = {
        "fl": [], "fr": [], "bl": [], "br": [],
        "other": [] 
    }
    
    # 1. Load Data
    names = []
    descriptors = []
    
    with h5py.File(global_features_path, 'r') as f:
        # We load all to memory for simplicity (assuming < 10k images)
        for name in f.keys():
            # hloc global features are groups containing 'global_descriptor'
            if 'global_descriptor' in f[name]:
                data = f[name]['global_descriptor'].__array__()
            else:
                # Fallback if it's a direct dataset (older hloc versions?)
                data = f[name].__array__()
            
            # hloc NetVLAD descriptors are typically shape (D,) or (1, D)
            if data.ndim == 2:
                data = data.squeeze(0)
            names.append(name)
            descriptors.append(data)
            
    if not names:
        print("No global features found.")
        return []
        
    # Stack
    descriptors = torch.from_numpy(np.stack(descriptors)).to(device)
    # L2 Normalize
    descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)
    
    # 2. Assign to buckets
    name_to_idx = {n: i for i, n in enumerate(names)}
    
    for name in names:
        stem = Path(name).stem
        assigned = False
        for suffix in ["_fl", "_fr", "_bl", "_br"]:
            if stem.endswith(suffix):
                buckets[suffix.replace("_","")].append(name)
                assigned = True
                break
        if not assigned:
            buckets["other"].append(name)

    all_pairs = set()
    
    # 3. Match within buckets
    for tag, bucket_names in buckets.items():
        if tag == "other":
            continue # User requested only 4 labelled buckets to be matched
            
        n_imgs = len(bucket_names)
        if n_imgs < 2:
            continue
            
        print(f"Matching bucket '{tag}': {n_imgs} images")
        
        # Get indices
        indices = [name_to_idx[n] for n in bucket_names]
        bucket_descs = descriptors[indices] # (B, D)
        
        # Sim matrix: B x B
        sim = torch.matmul(bucket_descs, bucket_descs.t())
        
        # Top K
        k = min(num_matched, n_imgs - 1)
        if k <= 0: continue
        
        # Set diagonal to -inf to ignore self
        sim.fill_diagonal_(-float('inf'))
        
        topk_scores, topk_indices = torch.topk(sim, k, dim=1)
        
        topk_indices = topk_indices.cpu().numpy()
        
        for i_local, row in enumerate(topk_indices):
            name_1 = bucket_names[i_local]
            for j_local in row:
                name_2 = bucket_names[j_local]
                
                # Sort to ensure uniqueness
                p = tuple(sorted((name_1, name_2)))
                all_pairs.add(p)
                
    print(f"Generated {len(all_pairs)} bucketed pairs.")
    
    # Write to file
    with open(output_pairs_path, "w") as f:
        for p1, p2 in sorted(all_pairs):
            f.write(f"{p1} {p2}\n")
            
    return all_pairs

def generate_bucketed_exhaustive_pairs(images_path, output_pairs_path):
    print(f"Generating bucketed EXHAUSTIVE pairs from {images_path}...")
    
    buckets = {
        "fl": [], "fr": [], "bl": [], "br": [],
        "other": [] 
    }
    
    # 1. List Images
    image_list = sorted([p.name for p in images_path.iterdir() if p.is_file()])
    
    # 2. Assign to buckets
    for name in image_list:
        stem = Path(name).stem
        assigned = False
        for suffix in ["_fl", "_fr", "_bl", "_br"]:
            if stem.endswith(suffix):
                buckets[suffix.replace("_","")].append(name)
                assigned = True
                break
        if not assigned:
            buckets["other"].append(name)

    all_pairs = set()
    
    # 3. Match within buckets EXHAUSTIVELY
    for tag, bucket_names in buckets.items():
        if tag == "other":
            continue # User requested only 4 labelled buckets to be matched
            
        n_imgs = len(bucket_names)
        if n_imgs < 2:
            continue
            
        print(f"Matching bucket '{tag}': {n_imgs} images (Exhaustive)")
        
        # itertools.combinations
        for i in range(len(bucket_names)):
            for j in range(i + 1, len(bucket_names)):
                p = (bucket_names[i], bucket_names[j])
                all_pairs.add(p)
                
    print(f"Generated {len(all_pairs)} bucketed exhaustive pairs.")
    
    # Write to file
    with open(output_pairs_path, "w") as f:
        for p1, p2 in sorted(all_pairs):
            f.write(f"{p1} {p2}\n")
            
    return all_pairs
