"""Search and evaluate frozen YOLO26m + DINOv3 image-level fusion."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO26m and DINOv3 score fusion.")
    parser.add_argument("--yolo-scores", type=Path, required=True)
    parser.add_argument("--dino-scores", type=Path, required=True)
    parser.add_argument("--split", choices=("calib", "holdout"), default="calib")
    parser.add_argument("--rule", choices=("or", "and"), default=None)
    parser.add_argument("--yolo-threshold", type=float, default=None)
    parser.add_argument("--dino-threshold", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_score_file(path: Path, score_column: str) -> tuple[dict[str, int], dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    labels: dict[str, int] = {}
    scores: dict[str, float] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            name = row["image_name"]
            if name in labels:
                raise RuntimeError(f"Duplicate image in {path}: {name}")
            labels[name] = int(row["label"])
            scores[name] = float(row[score_column])
    if not labels:
        raise RuntimeError(f"Empty score file: {path}")
    return labels, scores


def metrics(
    labels: dict[str, int],
    yolo_scores: dict[str, float],
    dino_scores: dict[str, float],
    rule: str,
    yolo_threshold: float,
    dino_threshold: float,
) -> dict[str, float | int | str]:
    tp = fp = fn = tn = 0
    for name, truth in labels.items():
        yolo_fire = yolo_scores[name] >= yolo_threshold
        dino_fire = dino_scores[name] >= dino_threshold
        predicted = (yolo_fire or dino_fire) if rule == "or" else (yolo_fire and dino_fire)
        if truth == 1 and predicted:
            tp += 1
        elif truth == 0 and predicted:
            fp += 1
        elif truth == 1:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "rule": rule,
        "yolo_threshold": float(yolo_threshold),
        "dino_threshold": float(dino_threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def main() -> None:
    args = parse_args()
    yolo_labels, yolo_scores = load_score_file(args.yolo_scores, "max_fire_confidence")
    dino_labels, dino_scores = load_score_file(args.dino_scores, "score")
    if set(yolo_labels) != set(dino_labels):
        raise RuntimeError("YOLO and DINO score files do not contain the same image names")
    if any(yolo_labels[name] != dino_labels[name] for name in yolo_labels):
        raise RuntimeError("YOLO and DINO score files disagree on labels")
    labels = yolo_labels
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.split == "calib":
        thresholds = np.linspace(0.01, 0.99, 99)
        scan = [
            metrics(labels, yolo_scores, dino_scores, rule, float(yolo_threshold), float(dino_threshold))
            for rule in ("or", "and")
            for yolo_threshold in thresholds
            for dino_threshold in thresholds
        ]
        scan.sort(key=lambda row: (float(row["f1"]), float(row["recall"]), float(row["precision"])), reverse=True)
        scan_path = args.output_dir / "yolo_dino_fusion_calib_scan.csv"
        with scan_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(scan[0]))
            writer.writeheader()
            writer.writerows(scan)
        summary = {
            "split": "calib",
            "images": len(labels),
            "positive_images": sum(labels.values()),
            "negative_images": len(labels) - sum(labels.values()),
            "yolo_scores": str(args.yolo_scores.resolve()),
            "dino_scores": str(args.dino_scores.resolve()),
            "scan_csv": str(scan_path.resolve()),
            "best_calib_rule": scan[0],
            "top_candidates": scan[:20],
            "holdout_used": False,
            "warning": "Freeze the selected rule and thresholds before one-shot holdout evaluation.",
        }
        summary_path = args.output_dir / "yolo_dino_fusion_calib_summary.json"
    else:
        if args.rule is None or args.yolo_threshold is None or args.dino_threshold is None:
            raise ValueError("holdout requires --rule, --yolo-threshold, and --dino-threshold")
        result = metrics(labels, yolo_scores, dino_scores, args.rule, args.yolo_threshold, args.dino_threshold)
        summary = {
            "split": "holdout",
            "images": len(labels),
            "positive_images": sum(labels.values()),
            "negative_images": len(labels) - sum(labels.values()),
            "yolo_scores": str(args.yolo_scores.resolve()),
            "dino_scores": str(args.dino_scores.resolve()),
            "frozen_rule": result,
            "holdout_used": True,
            "warning": "Holdout was evaluated once with the fusion rule frozen from calib.",
        }
        summary_path = args.output_dir / "yolo_dino_fusion_holdout_summary.json"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
