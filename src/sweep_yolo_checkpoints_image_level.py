"""Sweep selected YOLO checkpoints using the competition's image-level F1 metric."""
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np


def score(labels: dict[str, int], scores: dict[str, float], threshold: float) -> dict[str, float | int]:
    predicted = {name: value >= threshold for name, value in scores.items()}
    tp = sum(labels[name] == 1 and predicted[name] for name in labels)
    fp = sum(labels[name] == 0 and predicted[name] for name in labels)
    fn = sum(labels[name] == 1 and not predicted[name] for name in labels)
    tn = sum(labels[name] == 0 and not predicted[name] for name in labels)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": threshold, "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\fire_detection_clean\data\grouped_official_v1"))
    parser.add_argument("--split", choices=("calib",), default="calib")
    parser.add_argument("--epochs", nargs="+", type=int, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--predict-conf", type=float, default=0.01)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = args.data_root / "manifest.csv"
    with manifest.open("r", newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == args.split]
    labels = {row["image_name"]: int(row["image_level_label"]) for row in rows}
    paths = {row["image_name"]: Path(row["source_path"]) for row in rows}
    if not labels:
        raise RuntimeError(f"No {args.split} rows found in {manifest}")
    from ultralytics import YOLO
    import torch

    results = []
    names = list(labels)
    source = [str(paths[name]) for name in names]
    for epoch in sorted(set(args.epochs)):
        weights = args.run_dir / "weights" / f"epoch{epoch}.pt"
        if not weights.is_file():
            raise FileNotFoundError(weights)
        print(f"checkpoint epoch={epoch} start", flush=True)
        model = YOLO(str(weights))
        scores: dict[str, float] = {}
        predictions = model.predict(
            source=source,
            imgsz=args.imgsz,
            batch=args.batch,
            conf=args.predict_conf,
            iou=args.iou,
            device=args.device,
            half=args.half,
            stream=True,
            verbose=False,
        )
        for index, result in enumerate(predictions):
            boxes = result.boxes
            fire_scores = []
            if boxes is not None and len(boxes):
                fire_scores = [
                    float(boxes.conf[j].item())
                    for j in range(len(boxes))
                    if int(boxes.cls[j].item()) == 0
                ]
            scores[names[index]] = max(fire_scores, default=0.0)
        scan = [score(labels, scores, float(threshold)) for threshold in np.linspace(0.01, 0.99, 99)]
        best = max(scan, key=lambda row: (row["f1"], row["recall"], row["precision"]))
        row = {"epoch": epoch, **best}
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        del predictions, model, scores
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results.sort(key=lambda row: (row["f1"], row["recall"], row["precision"]), reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"split": args.split, "run_dir": str(args.run_dir.resolve()), "results": results, "holdout_used": False}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
