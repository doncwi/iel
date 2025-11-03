#!/usr/bin/env python3
"""
embedlab.py — Minimal image embedding and search CLI using FastEmbed

Commands:
  1. Embed training images:
     python embedlab.py embed --images-dir ./assets/images --out ./index

  2. Search similar images:
     python embedlab.py search --index ./index --query-dir ./assets/queries --k 5 --json
"""
import time
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from fastembed import ImageEmbedding
from sklearn.metrics.pairwise import cosine_similarity
from PIL import ImageStat
import math

# -------------------------------------------
# Utility functions
# -------------------------------------------


def list_images(directory):
    """Return list of valid image file paths in directory."""
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    return sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(exts)
    ])


def save_index(out_dir, paths, embeddings):
    """Save embeddings and corresponding paths."""
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "embeddings.npy"), embeddings)
    with open(os.path.join(out_dir, "paths.json"), "w") as f:
        json.dump(paths, f)
    print(f"✅ Saved index to {out_dir}/")


def load_index(index_dir):
    """Load embeddings and paths from index directory."""
    emb_path = os.path.join(index_dir, "embeddings.npy")
    path_path = os.path.join(index_dir, "paths.json")
    if not os.path.exists(emb_path) or not os.path.exists(path_path):
        raise FileNotFoundError("Index not found or incomplete.")
    embeddings = np.load(emb_path)
    with open(path_path, "r") as f:
        paths = json.load(f)
    return paths, embeddings


def image_entropy(img):
    """Compute Shannon entropy of a PIL image (0 = uniform, ~8 = rich)."""
    hist = img.convert("L").histogram()
    hist_size = float(sum(hist))
    hist = [h / hist_size for h in hist if h > 0]
    return -sum(p * math.log2(p) for p in hist)


def is_image_valid(path, entropy_threshold=1.5, std_threshold=3):
    """
    Check for corruption, blank, or low-entropy images.
    Returns (is_valid, reason)
    """
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        return False, f"corrupted ({e})"

    stat = ImageStat.Stat(img)
    mean = sum(stat.mean) / 3
    stddev = sum(stat.stddev) / 3

    if stddev < std_threshold:
        return False, f"low contrast (std={stddev:.2f})"

    ent = image_entropy(img)
    if ent < entropy_threshold:
        return False, f"low entropy (H={ent:.2f})"

    return True, None

# -------------------------------------------
# Embedding Command
# -------------------------------------------


def cmd_embed(args):
    start = time.time()
    image_dir = args.images_dir
    out_dir = args.out

    paths = list_images(image_dir)
    if not paths:
        print(f"❌ No images found in {image_dir}")
        return

    embedder = ImageEmbedding(model_name="Qdrant/clip-ViT-B-32-vision")
    embeddings = []
    valid_paths = []

    print(f"🧠 Embedding {len(paths)} images from {image_dir} ...")
    skipped = []

    count = 0
    for path in tqdm(paths):
        valid, reason = is_image_valid(path)
        if not valid:
            skipped.append((path, reason))
            continue

        image = Image.open(path).convert("RGB")
        emb = next(embedder.embed([image]))
        embeddings.append(emb)
        valid_paths.append(path)
        count = count + 1

    if not embeddings:
        print("❌ No valid images to embed.")
        return

    embeddings = np.vstack(embeddings)
    save_index(out_dir, valid_paths, embeddings)

    if skipped:
        print(f"⚠️ Skipped {len(skipped)} bad images:")
        for p, r in skipped[:10]:
            print(f"  - {p}: {r}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped)-10} more")

    end = time.time()
    print(f"Embeds took {end - start:.4f} seconds to go over {count} files")


# -------------------------------------------
# Search Command
# -------------------------------------------


def cmd_search(args):
    start = time.time()

    """Search top-K similar images for query images."""
    index_dir = args.index
    query_dir = args.query_dir
    k = args.k
    as_json = args.json

    # Load index
    print(f"📂 Loading index from {index_dir} ...")
    paths, embeddings = load_index(index_dir)

    query_paths = list_images(query_dir)
    if not query_paths:
        print(f"❌ No query images found in {query_dir}")
        return

    embedder = ImageEmbedding(model_name="Qdrant/clip-ViT-B-32-vision")

    results = []

    print(
        f"🔍 Searching top-{k} similar images for {len(query_paths)} queries ...")
    for qpath in tqdm(query_paths):
        qimg = Image.open(qpath).convert("RGB")
        qemb = next(embedder.embed([qimg]))
        sims = cosine_similarity([qemb], embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:k]
        top_results = [
            {"path": paths[i], "score": float(sims[i])} for i in top_idx
        ]
        results.append({
            "query": qpath,
            "results": top_results
        })

    if as_json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            print(f"\nQuery: {r['query']}")
            for res in r["results"]:
                print(f"  {res['path']}  ({res['score']:.4f})")
    end = time.time()
    print(f"Search took {end - start:.4f} seconds")


# -------------------------------------------
# Analyze Command
# -------------------------------------------

def cmd_analyze(args):
    """Find near-duplicate image groups and most isolated anomalies."""
    index_dir = args.index
    dup_threshold = args.dup_threshold
    anomaly_top = args.anomaly_top
    as_json = args.json

    print(f"📂 Loading index from {index_dir} ...")
    paths, embeddings = load_index(index_dir)
    n = len(paths)
    print(f"🧩 Loaded {n} embeddings")

    # --- Compute cosine similarity matrix (symmetric) ---
    print("📈 Computing cosine similarities...")
    sims = cosine_similarity(embeddings)
    np.fill_diagonal(sims, 1.0)

    # --- Find duplicate groups ---
    visited = set()
    duplicate_groups = []
    for i in range(n):
        if i in visited:
            continue
        dup_indices = set(np.where(sims[i] >= dup_threshold)[0])
        dup_indices.discard(i)
        if dup_indices:
            group = [paths[i]] + [paths[j] for j in dup_indices]
            duplicate_groups.append(sorted(group))
            visited.update(dup_indices)
            visited.add(i)

    # --- Find anomalies ---
    print("🔍 Detecting anomalies...")
    k = min(10, n - 1)  # K for KNN density
    dists = 1 - sims
    knn_mean_dist = np.mean(np.sort(dists, axis=1)[:, 1:k+1], axis=1)
    top_anomaly_idx = np.argsort(knn_mean_dist)[::-1][:anomaly_top]
    anomalies = [paths[i] for i in top_anomaly_idx]

    result = {
        "duplicate_groups": duplicate_groups,
        "anomalies": anomalies
    }

    if as_json:
        print(json.dumps(result, indent=2))
    else:
        print("\n🧍 Duplicate groups:")
        for g in duplicate_groups:
            print("  -", ", ".join(g))
        print("\n🚨 Anomalies:")
        for a in anomalies:
            print("  -", a)


# -------------------------------------------
# Main CLI Entrypoint
# -------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Fast image embedding + similarity search CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- embed command ---
    p_embed = subparsers.add_parser(
        "embed", help="Embed images and save index")
    p_embed.add_argument("--images-dir", required=True,
                         help="Path to directory with images")
    p_embed.add_argument("--out", required=True,
                         help="Output directory for index")
    p_embed.set_defaults(func=cmd_embed)

    # --- search command ---
    p_search = subparsers.add_parser(
        "search", help="Search for similar images")
    p_search.add_argument("--index", required=True,
                          help="Path to existing index directory")
    p_search.add_argument("--query-dir", required=True,
                          help="Directory with query images")
    p_search.add_argument("--k", type=int, default=5,
                          help="Top-K results (default: 5)")
    p_search.add_argument("--json", action="store_true", help="Output as JSON")
    p_search.set_defaults(func=cmd_search)

    # --- analyze command ---
    p_analyze = subparsers.add_parser(
        "analyze", help="Analyze index for duplicates and anomalies")
    p_analyze.add_argument("--index", required=True,
                           help="Path to existing index directory")
    p_analyze.add_argument("--dup-threshold", type=float, default=0.92,
                           help="Cosine similarity threshold for near-duplicates")
    p_analyze.add_argument("--anomaly-top", type=int,
                           default=8, help="Top-N anomalies to report")
    p_analyze.add_argument("--json", action="store_true",
                           help="Output as JSON")
    p_analyze.set_defaults(func=cmd_analyze)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
