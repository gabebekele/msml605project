import os
import sys
import csv
import time
import argparse

import yaml
import numpy as np
import tensorflow as tf
import torch

# Make project root importable
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.embedding import build_image_cache, MODEL
from m1.similarity import cosine_similarity
from src.thresholding import apply_threshold


DEFAULT_THRESHOLD = 0.30612244897959173


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def read_pairs_csv(pairs_path, limit):
    pairs = []

    with open(pairs_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            left = row.get("left_path")
            right = row.get("right_path")
            label = row.get("label", "")

            if left is None or right is None:
                raise ValueError("CSV must contain left_path and right_path columns.")

            pairs.append({
                "left_path": left,
                "right_path": right,
                "label": label
            })

            if len(pairs) >= limit:
                break

    return pairs


def preprocess_image(path_str, image_cache, target_shape=(160, 160), normalization=None):
    relative_path = path_str.replace("\\", "/")

    if relative_path not in image_cache:
        raise FileNotFoundError(f"Image not found for key: {relative_path}")

    img_array = image_cache[relative_path]

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    img_tensor = tf.image.resize(img_tensor, target_shape)

    if normalization == "z-score":
        mean, variance = tf.nn.moments(img_tensor, axes=[0, 1, 2])
        img_tensor = (img_tensor - mean) / tf.sqrt(variance + 1e-7)
    elif normalization == "min-max":
        img_tensor = img_tensor / 255.0
    else:
        img_tensor = img_tensor / 255.0

    img_array = img_tensor.numpy().astype(np.float32)

    # Convert HWC image format to BCHW PyTorch format
    img_tensor_pt = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)

    return img_tensor_pt


def generate_embedding(img_tensor_pt):
    with torch.no_grad():
        embedding = MODEL(img_tensor_pt)

    return embedding.squeeze(0).numpy()


def profile_pair(left_path, right_path, image_cache, threshold, normalization=None):
    total_start = time.perf_counter()

    preprocess_start = time.perf_counter()

    left_img = preprocess_image(
        left_path,
        image_cache=image_cache,
        normalization=normalization
    )

    right_img = preprocess_image(
        right_path,
        image_cache=image_cache,
        normalization=normalization
    )

    preprocess_time = time.perf_counter() - preprocess_start

    embedding_start = time.perf_counter()

    left_embedding = generate_embedding(left_img)
    right_embedding = generate_embedding(right_img)

    embedding_time = time.perf_counter() - embedding_start

    scoring_start = time.perf_counter()

    score = cosine_similarity(
        left_embedding.reshape(1, -1),
        right_embedding.reshape(1, -1)
    )[0]

    prediction = apply_threshold(
        scores=[score],
        threshold=threshold,
        higher_is_same=True
    )[0]

    confidence = abs(float(score) - float(threshold))

    scoring_time = time.perf_counter() - scoring_start

    total_time = time.perf_counter() - total_start

    return {
        "score": float(score),
        "prediction": int(prediction),
        "confidence": float(confidence),
        "preprocess_seconds": float(preprocess_time),
        "embedding_seconds": float(embedding_time),
        "scoring_seconds": float(scoring_time),
        "end_to_end_seconds": float(total_time)
    }


def summarize(rows, batch_size):
    preprocess = np.array([r["preprocess_seconds"] for r in rows])
    embedding = np.array([r["embedding_seconds"] for r in rows])
    scoring = np.array([r["scoring_seconds"] for r in rows])
    total = np.array([r["end_to_end_seconds"] for r in rows])

    total_wall_time = np.sum(total)
    if total_wall_time > 0:
        throughput = len(rows) / total_wall_time
    else:
        total_wall_time = 0.0

    return {
        "batch_size": batch_size,
        "num_pairs": len(rows),
        "avg_preprocess": np.mean(preprocess),
        "avg_embedding": np.mean(embedding),
        "avg_scoring": np.mean(scoring),
        "avg_total": np.mean(total),
        "p95_total": np.percentile(total, 95),
        "throughput": throughput
    }


def main():
    parser = argparse.ArgumentParser(
        description="Profile preprocessing, embedding, and scoring latency."
    )

    parser.add_argument("--config", default="configs/m1.yaml")
    parser.add_argument("--pairs", default="outputs/pairs/val.csv")
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 5, 10, 25])
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--normalization", default=None, choices=[None, "z-score", "min-max"])

    args = parser.parse_args()

    config = load_config(args.config)
    lfw_dir = config["data_dir"]

    print("Loading deterministic LFW image cache...")
    image_cache = build_image_cache(lfw_dir)

    max_pairs_needed = max(args.batch_sizes)
    all_pairs = read_pairs_csv(args.pairs, max_pairs_needed)

    if len(all_pairs) < max_pairs_needed:
        raise ValueError(
            f"Requested {max_pairs_needed} pairs, but only found {len(all_pairs)}."
        )

    print("\nMilestone 4 Profiling Report Output\n")
   
    print(f"Threshold: {args.threshold}")
    print(f"Normalization: {args.normalization}")
    print("Embedding model: InceptionResnetV1 pretrained on VGGFace2")
    print("Scoring method: Cosine similarity")
    print("Hardware profile: CPU baseline\n")

    summaries = []

    for batch_size in args.batch_sizes:
        batch_pairs = all_pairs[:batch_size]
        rows = []

        for pair in batch_pairs:
            result = profile_pair(
                pair["left_path"],
                pair["right_path"],
                image_cache,
                threshold=args.threshold,
                normalization=args.normalization
            )
            rows.append(result)

        summary = summarize(rows, batch_size)
        summaries.append(summary)

    print("\nBatch-Size Sensitivity Results")
    print(
        "batch_size | num_pairs | avg_preprocess_s | avg_embedding_s | "
        "avg_scoring_s | avg_total_s | p95_total_s | throughput_pairs_per_s"
    )
    print("-" * 150)

    for s in summaries:
        print(
            f"{s['batch_size']:>10} | "
            f"{s['num_pairs']:>9} | "
            f"{s['avg_preprocess']:.6f} | "
            f"{s['avg_embedding']:.6f} | "
            f"{s['avg_scoring']:.6f} | "
            f"{s['avg_total']:.6f} | "
            f"{s['p95_total']:.6f} | "
            f"{s['throughput']:.6f}"
        )



if __name__ == "__main__":
    main()