"""Evaluate image-level fire metrics on a frozen split protocol."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO image-level fire metrics.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--data-root", type=Path,
        default=Path(r"C:\fire_detection_clean\data\grouped_official_v1"),
    )
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument(
        "--predict-conf", type=float, default=0.01,
        help="Minimum detector confidence during scoring; thresholds start at 0.01.",
    )
    parser.add_argument("--half", action="store_true", help="Use FP16 inference on CUDA.")
    parser.add_argument("--augment", action="store_true", help="Use Ultralytics test-time augmentation.")
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="0")
    parser.add_argument("--split", choices=("calib", "holdout"), default="calib")
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Frozen image-level threshold; required for holdout evaluation.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=None,
    )
    return parser.parse_args()


def metrics(labels: dict[str, int], scores: dict[str, float], threshold: float) -> dict[str, float | int]:
    tp = fp = fn = tn = 0
    for name, truth in labels.items():
        predicted = scores[name] >= threshold
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
        "threshold": float(threshold),
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
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)

    if args.split == "holdout" and args.threshold is None:
        raise ValueError("--threshold is required for holdout evaluation")

    if args.output_dir is None:
        args.output_dir = Path(rf"C:\fire_detection_clean\runs\image_level_{args.split}")

    manifest_path = args.data_root / "manifest.csv"
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == args.split]
    if not rows:
        raise RuntimeError(f"No {args.split} rows found in {manifest_path}")

    labels = {row["image_name"]: int(row["image_level_label"]) for row in rows}
    paths = {row["image_name"]: Path(row["source_path"]) for row in rows}
    if len(labels) != len(rows) or set(paths) != set(labels):
        raise RuntimeError("Duplicate or inconsistent calib manifest entries")
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing calib images: {missing[:5]}")

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    # Keep the original batched streaming inference protocol. Ultralytics may
    # expose generic image names for list input, so pair results by order.
    split_names = list(labels)
    scores: dict[str, float] = {}
    results = model.predict(
        source=[str(paths[name]) for name in labels],
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.predict_conf,
        iou=args.iou,
        device=args.device,
        half=args.half,
        augment=args.augment,
        stream=True,
        verbose=False,
    )
    for index, result in enumerate(results):
        if index >= len(split_names):
            raise RuntimeError(f"Received more predictions than {args.split} images")
        name = split_names[index]
        boxes = result.boxes
        fire_scores = []
        if boxes is not None and len(boxes):
            fire_scores = [
                float(boxes.conf[index].item())
                for index in range(len(boxes))
                if int(boxes.cls[index].item()) == 0
            ]
        scores[name] = max(fire_scores, default=0.0)

    missing_predictions = sorted(set(labels) - set(scores))
    if missing_predictions:
        raise RuntimeError(f"Incomplete predictions: {missing_predictions[:10]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    score_path = args.output_dir / f"{args.split}_image_scores.csv"
    with score_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_name", "label", "max_fire_confidence"])
        writer.writeheader()
        for name in sorted(labels):
            writer.writerow({
                "image_name": name,
                "label": labels[name],
                "max_fire_confidence": scores[name],
            })

    summary = {
        "split": args.split,
        "images": len(labels),
        "positive_images": sum(labels.values()),
        "negative_images": len(labels) - sum(labels.values()),
        "weights": str(args.weights.resolve()),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "predict_conf": args.predict_conf,
        "half": args.half,
        "augment": args.augment,
        "iou": args.iou,
        "image_scores_csv": str(score_path.resolve()),
    }
    if args.split == "calib":
        thresholds = np.linspace(0.01, 0.99, 99)
        scan = [metrics(labels, scores, float(threshold)) for threshold in thresholds]
        table_path = args.output_dir / "calib_image_level_thresholds.csv"
        with table_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(scan[0]))
            writer.writeheader()
            writer.writerows(scan)
        best_f1 = max(scan, key=lambda row: (float(row["f1_diagnostic"]), float(row["recall"]), float(row["precision"])))
        summary["threshold_scan_csv"] = str(table_path.resolve())
        summary["best_f1_row_on_calib_diagnostic_only"] = best_f1
        summary["holdout_used"] = False
        summary["warning"] = "Calib metrics are for model/threshold development only; freeze all choices before one-shot holdout evaluation."
        summary_name = "calib_image_level_summary.json"
    else:
        final_metrics = metrics(labels, scores, float(args.threshold))
        summary["frozen_threshold"] = float(args.threshold)
        summary["metrics_at_frozen_threshold"] = final_metrics
        summary["holdout_used"] = True
        summary["warning"] = "Holdout was evaluated once with the threshold frozen from calib."
        summary_name = "holdout_image_level_summary.json"

    summary_path = args.output_dir / summary_name
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
