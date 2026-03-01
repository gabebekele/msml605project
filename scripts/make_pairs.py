import os
import csv
import yaml


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def write_pairs_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["left_path", "right_path", "label", "split"]
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def main(config_path: str):
    config = load_config(config_path)

    for split in ["train", "val", "test"]:
        pairs_path = os.path.join(
            config["outputs_dir"],
            "pairs",
            f"{split}.csv"
        )

        write_pairs_csv(pairs_path, rows=[])
        print(f"Wrote pairs -> {pairs_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    main(args.config)