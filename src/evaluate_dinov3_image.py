"""Evaluate a frozen DINOv3 image-level fire head on one split."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class LinearHead(nn.Module):
    def __init__(self, dimension: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Dropout(dropout),
            nn.Linear(dimension, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a frozen DINOv3 image classifier.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path(r"C:\fire_detection_clean\runs\dinov3_official_v1\best_head.pt"))
    parser.add_argument("--cache-dir", type=Path, default=Path(r"C:\AI\hf_cache\hub"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-name", default="holdout")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu; defaults to CUDA when available")
    parser.add_argument("--threshold", type=float, default=None, help="Frozen threshold; required for holdout")
    parser.add_argument("--scan-thresholds", action="store_true", help="Scan thresholds; use only on calib")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty manifest: {path}")
    for row in rows:
        row["label"] = int(row["label"])
        if not Path(str(row["image_path"])).is_file():
            raise FileNotFoundError(str(row["image_path"]))
    return rows


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted = scores >= threshold
    tp = int(np.sum((labels == 1) & predicted))
    fp = int(np.sum((labels == 0) & predicted))
    fn = int(np.sum((labels == 1) & ~predicted))
    tn = int(np.sum((labels == 0) & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": float(threshold), "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main() -> None:
    args = parse_args()
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.split_name == "holdout" and args.threshold is None and not args.scan_thresholds:
        raise ValueError("--threshold is required for holdout evaluation")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(args.cache_dir.parent)
    os.environ["HF_HUB_CACHE"] = str(args.cache_dir)
    rows = load_rows(args.manifest)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_name = checkpoint["model"]
    print(f"device: {device}")
    print(f"loading frozen backbone: {model_name}")
    import timm
    from timm.data import create_transform, resolve_model_data_config

    backbone = timm.create_model(model_name, pretrained=True, num_classes=0).eval().to(device)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    transform = create_transform(**resolve_model_data_config(backbone), is_training=False)
    head = LinearHead(int(checkpoint["feature_dimension"]), float(checkpoint["dropout"])).to(device)
    head.load_state_dict(checkpoint["head_state_dict"])
    head.eval()

    labels: list[int] = []
    names: list[str] = []
    scores: list[float] = []
    for start in range(0, len(rows), args.batch_size):
        current = rows[start : start + args.batch_size]
        tensors = []
        for row in current:
            with Image.open(str(row["image_path"])) as image:
                tensors.append(transform(image.convert("RGB")))
        batch = torch.stack(tensors).to(device)
        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features = backbone(batch)
            else:
                features = backbone(batch)
            features = torch.nn.functional.normalize(features.float(), dim=1)
            batch_scores = torch.softmax(head(features), dim=1)[:, 1].cpu().numpy()
        labels.extend(int(row["label"]) for row in current)
        names.extend(str(row["image_name"]) for row in current)
        scores.extend(float(score) for score in batch_scores)
        print(f"evaluate {args.split_name}: {min(start + args.batch_size, len(rows))}/{len(rows)}")

    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    score_path = args.output_dir / f"{args.split_name}_dino_scores.csv"
    with score_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_name", "label", "score"])
        writer.writeheader()
        for name, label, score in zip(names, labels, scores):
            writer.writerow({"image_name": name, "label": label, "score": score})

    summary: dict[str, object] = {
        "split": args.split_name,
        "images": len(rows),
        "positive_images": int(labels_array.sum()),
        "negative_images": int(len(labels_array) - labels_array.sum()),
        "checkpoint": str(args.checkpoint.resolve()),
        "manifest": str(args.manifest.resolve()),
        "scores_csv": str(score_path.resolve()),
        "holdout_used": args.split_name == "holdout",
    }
    if args.scan_thresholds or args.split_name == "calib":
        scan = [metrics(labels_array, scores_array, float(t)) for t in np.linspace(0.01, 0.99, 99)]
        scan.sort(key=lambda row: (row["f1"], row["recall"], row["precision"]), reverse=True)
        scan_path = args.output_dir / f"{args.split_name}_dino_thresholds.csv"
        with scan_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(scan[0]))
            writer.writeheader()
            writer.writerows(scan)
        summary["threshold_scan_csv"] = str(scan_path.resolve())
        summary["best_metrics"] = scan[0]
        summary["holdout_used"] = False
    else:
        summary["frozen_metrics"] = metrics(labels_array, scores_array, float(args.threshold))
        summary["frozen_threshold"] = float(args.threshold)
    summary_path = args.output_dir / f"{args.split_name}_dino_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
