import os, json, yaml
import numpy as np
import tensorflow_datasets as tfds


def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(cfg_path):
    cfg = load_cfg(cfg_path)
    seed = int(cfg["seed"])
    data_dir = cfg["data_dir"]
    out_dir = cfg["outputs_dir"]

    tr = float(cfg.get("train_frac", 0.70))
    va = float(cfg.get("val_frac", 0.15))
    te = float(cfg.get("test_frac", 0.15))
    if not np.isclose(tr + va + te, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    os.makedirs(out_dir, exist_ok=True)

    # Load LFW from TFDS
    ds = tfds.load("lfw", split="train", data_dir=data_dir)

    # deterministic list
    records = []
    per_id_index = {}

    for ex in tfds.as_numpy(ds):
        ident = ex["label"].decode("utf-8")  # label = name in TFDS lfw
        per_id_index[ident] = per_id_index.get(ident, 0) + 1
        fname = f"{ident}_{per_id_index[ident]:04d}.jpg"
        records.append((ident, fname))


    # Deterministic identity split
    identities = sorted({ident for ident, _ in records})
    rng = np.random.default_rng(seed)
    rng.shuffle(identities)

    n = len(identities)
    n_tr = int(tr * n)
    n_va = int(va * n)
    tr_ids = set(identities[:n_tr])
    va_ids = set(identities[n_tr:n_tr + n_va])
    te_ids = set(identities[n_tr + n_va:])

    # manifest counts
    counts = {
        "train": {"images": 0, "identities": len(tr_ids)},
        "val":   {"images": 0, "identities": len(va_ids)},
        "test":  {"images": 0, "identities": len(te_ids)},
    }
    for ident, _ in records:
        if ident in tr_ids:
            counts["train"]["images"] += 1
        elif ident in va_ids:
            counts["val"]["images"] += 1
        else:
            counts["test"]["images"] += 1

    manifest = {
        "seed": seed,
        "split_policy": cfg["split_policy"],
        "counts": counts,
        "data_source": {"how": "TFDS dataset: lfw (split='train')", "cache_dir": data_dir},
    }

    path = os.path.join(out_dir, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"Wrote manifest -> {path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    main(args.config)
