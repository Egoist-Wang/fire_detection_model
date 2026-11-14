"""Search and evaluate an image-level fusion of two clean YOLO score files."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO s/m image-score fusion.")
    parser.add_argument("--s-scores", type=Path, required=True)
    parser.add_argument("--m-scores", type=Path, required=True)
    parser.add_argument("--split", choices=("calib", "holdout"), default="calib")
    parser.add_argument("--rule", choices=("or", "and"), default=None)
    parser.add_argument("--s-threshold", type=float, default=None)
    parser.add_argument("--m-threshold", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_scores(path: Path) -> tuple[dict[str, int], dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    labels: dict[str, int] = {}
    scores: dict[str, float] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = row["image_name"]
            if name in labels:
                raise RuntimeError(f"Duplicate image in {path}: {name}")
            labels[name] = int(row["label"])
            scores[name] = float(row["max_fire_confidence"])
    if not labels:
        raise RuntimeError(f"Empty score file: {path}")
    return labels, scores


def validate_pair(
    s_labels: dict[str, int],
    s_scores: dict[str, float],
    m_labels: dict[str, int],
    m_scores: dict[str, float],
) -> dict[str, int]:
    if set(s_labels) != set(m_labels):
        raise RuntimeError("The two score files do not contain the same image names")
    if any(s_labels[name] != m_labels[name] for name in s_labels):
        raise RuntimeError("The two score files disagree on image labels")
    return s_labels


def metrics(
    labels: dict[str, int],
    s_scores: dict[str, float],
    m_scores: dict[str, float],
    rule: str,
    s_threshold: float,
    m_threshold: float,
) -> dict[str, float | int | str]:
    tp = fp = fn = tn = 0
    for name, truth in labels.items():
        s_fire = s_scores[name] >= s_threshold
        m_fire = m_scores[name] >= m_threshold
        predicted = (s_fire or m_fire) if rule == "or" else (s_fire and m_fire)
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
        "s_threshold": float(s_threshold),
        "m_threshold": float(m_threshold),
        "precision": precision,
        "recall": recall,
        "f1_diagnostic": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def main() -> None:
    args = parse_args()
    s_labels, s_scores = load_scores(args.s_scores)
    m_labels, m_scores = load_scores(args.m_scores)
    labels = validate_pair(s_labels, s_scores, m_labels, m_scores)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.split == "calib":
        thresholds = np.linspace(0.01, 0.99, 99)
        scan = [
            metrics(labels, s_scores, m_scores, rule, float(s_threshold), float(m_threshold))
            for rule in ("or", "and")
            for s_threshold in thresholds
            for m_threshold in thresholds
        ]
        scan.sort(key=lambda row: (float(row["f1_diagnostic"]), float(row["recall"]), float(row["precision"])), reverse=True)
        scan_path = args.output_dir / "yolo_sm_fusion_calib_scan.csv"
        with scan_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(scan[0]))
            writer.writeheader()
            writer.writerows(scan)
        summary = {
            "split": "calib",
            "images": len(labels),
            "positive_images": sum(labels.values()),
            "negative_images": len(labels) - sum(labels.values()),
            "s_scores": str(args.s_scores.resolve()),
            "m_scores": str(args.m_scores.resolve()),
            "scan_csv": str(scan_path.resolve()),
            "best_f1_rule_on_calib_diagnostic_only": scan[0],
            "top_candidates": scan[:20],
            "holdout_used": False,
            "warning": "Freeze the selected rule and thresholds before evaluating holdout.",
        }
        summary_path = args.output_dir / "yolo_sm_fusion_calib_summary.json"
    else:
        if args.rule is None or args.s_threshold is None or args.m_threshold is None:
            raise ValueError("holdout requires --rule, --s-threshold, and --m-threshold")
        result = metrics(labels, s_scores, m_scores, args.rule, args.s_threshold, args.m_threshold)
        summary = {
            "split": "holdout",
            "images": len(labels),
            "positive_images": sum(labels.values()),
            "negative_images": len(labels) - sum(labels.values()),
            "s_scores": str(args.s_scores.resolve()),
            "m_scores": str(args.m_scores.resolve()),
            "frozen_rule": result,
            "holdout_used": True,
            "warning": "Holdout was evaluated once with the fusion rule frozen from calib.",
        }
        summary_path = args.output_dir / "yolo_sm_fusion_holdout_summary.json"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
