import os
import sys
import csv
import yaml
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from m1.similarity import cosine_similarity
from embedding import embed_pair
from thresholding import apply_threshold, compute_confusion_matrix
from thresholding import compute_accuracy, compute_precision, compute_recall
from thresholding import compute_f1_score, compute_balanced_accuracy
from validation import validate_pairs_file, validate_threshold, validate_scores_and_labels
from tracking import save_json, build_run_record, save_run


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_pairs(path):
    pair_rows = []

    with open(path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            pair_rows.append(row)

    return pair_rows


def compute_similarity_scores(pair_rows):
    left_vectors = []
    right_vectors = []
    true_labels = []

    for row in pair_rows:
        left_vector, right_vector = embed_pair(row["left_path"], row["right_path"])
        left_vectors.append(left_vector)
        right_vectors.append(right_vector)
        true_labels.append(int(row["label"]))

    left_vectors = np.array(left_vectors)
    right_vectors = np.array(right_vectors)

    scores = cosine_similarity(left_vectors, right_vectors)

    validate_scores_and_labels(scores, true_labels)

    return scores, np.array(true_labels)


def compute_metrics_at_threshold(scores, true_labels, threshold, higher_is_same=True):
    validate_threshold(threshold)

    predicted_labels = apply_threshold(
        scores=scores,
        threshold=threshold,
        higher_is_same=higher_is_same
    )

    confusion = compute_confusion_matrix(true_labels, predicted_labels)

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(compute_accuracy(confusion)),
        "precision": float(compute_precision(confusion)),
        "recall": float(compute_recall(confusion)),
        "f1": float(compute_f1_score(confusion)),
        "balanced_accuracy": float(compute_balanced_accuracy(confusion)),
        "tp": confusion["tp"],
        "tn": confusion["tn"],
        "fp": confusion["fp"],
        "fn": confusion["fn"]
    }

    return metrics


def run_threshold_sweep(scores, true_labels, thresholds, higher_is_same=True):
    sweep_results = []

    for threshold in thresholds:
        metrics = compute_metrics_at_threshold(
            scores=scores,
            true_labels=true_labels,
            threshold=float(threshold),
            higher_is_same=higher_is_same
        )
        sweep_results.append(metrics)

    return sweep_results


def select_best_threshold(sweep_results, metric_name="balanced_accuracy"):
    if len(sweep_results) == 0:
        raise ValueError("Sweep results cannot be empty")

    best_result = max(sweep_results, key=lambda result: result[metric_name])
    return best_result


def evaluate_split(config, split_name, threshold=None, pair_version="baseline"):
    pairs_path = os.path.join(config["outputs_dir"], "pairs", f"{split_name}.csv")
    validate_pairs_file(pairs_path)

    pair_rows = load_pairs(pairs_path)
    scores, true_labels = compute_similarity_scores(pair_rows)

    eval_dir = os.path.join(config["outputs_dir"], "eval")
    os.makedirs(eval_dir, exist_ok=True)

    scores_output_path = os.path.join(eval_dir, f"{split_name}_scores.json")
    save_json(scores_output_path, {
        "split": split_name,
        "num_pairs": len(pair_rows),
        "scores": [float(score) for score in scores],
        "labels": [int(label) for label in true_labels]
    })

    print(f"Saved {split_name} scores to {scores_output_path}")

    if threshold is None:
        threshold_values = np.linspace(-1.0, 1.0, 50)

        sweep_results = run_threshold_sweep(
            scores=scores,
            true_labels=true_labels,
            thresholds=threshold_values,
            higher_is_same=True
        )

        best_result = select_best_threshold(
            sweep_results=sweep_results,
            metric_name="balanced_accuracy"
        )

        sweep_output_path = os.path.join(eval_dir, f"{split_name}_sweep.json")
        selected_output_path = os.path.join(eval_dir, f"{split_name}_selected.json")

        save_json(sweep_output_path, sweep_results)
        save_json(selected_output_path, best_result)

        run_id = f"{split_name}_{pair_version}_sweep"
        run_record = build_run_record(
            run_id=run_id,
            split_name=split_name,
            pair_version=pair_version,
            threshold_info={
                "mode": "selected_from_sweep",
                "rule": "maximize balanced_accuracy",
                "selected_threshold": float(best_result["threshold"])
            },
            metrics=best_result,
            note=f"{split_name} threshold sweep and selected threshold"
        )
        run_output_path = save_run(config["outputs_dir"], run_id, run_record)

        print(f"Saved {split_name} threshold sweep to {sweep_output_path}")
        print(f"Saved selected {split_name} threshold to {selected_output_path}")
        print(f"Saved run log to {run_output_path}")

        return best_result["threshold"], best_result

    final_metrics = compute_metrics_at_threshold(
        scores=scores,
        true_labels=true_labels,
        threshold=float(threshold),
        higher_is_same=True
    )

    final_output_path = os.path.join(eval_dir, f"{split_name}_final.json")
    save_json(final_output_path, final_metrics)

    run_id = f"{split_name}_{pair_version}_final"
    run_record = build_run_record(
        run_id=run_id,
        split_name=split_name,
        pair_version=pair_version,
        threshold_info={
            "mode": "fixed_threshold",
            "rule": "provided_threshold",
            "selected_threshold": float(threshold)
        },
        metrics=final_metrics,
        note=f"{split_name} final evaluation at locked threshold"
    )
    run_output_path = save_run(config["outputs_dir"], run_id, run_record)

    print(f"Saved {split_name} final metrics to {final_output_path}")
    print(f"Saved run log to {run_output_path}")

    return float(threshold), final_metrics


def main(config_path):
    config = load_config(config_path)

    selected_threshold, validation_metrics = evaluate_split(
        config=config,
        split_name="val",
        threshold=None,
        pair_version="baseline"
    )

    _, test_metrics = evaluate_split(
        config=config,
        split_name="test",
        threshold=selected_threshold,
        pair_version="baseline"
    )

    summary = {
        "threshold_rule": "maximize balanced_accuracy on validation",
        "selected_validation_threshold": float(selected_threshold),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics
    }

    summary_output_path = os.path.join(config["outputs_dir"], "eval", "summary.json")
    save_json(summary_output_path, summary)

    print(f"Saved evaluation summary to {summary_output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    main(args.config)
