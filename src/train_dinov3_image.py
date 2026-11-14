"""Train a linear image-level fire head on frozen DINOv3 features."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset


class LinearHead(nn.Module):
    def __init__(self, dimension: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.LayerNorm(dimension), nn.Dropout(dropout), nn.Linear(dimension, 2))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty manifest: {path}")
    for row in rows:
        row["label"] = int(row["label"])
        if not Path(str(row["image_path"])).is_file():
            raise FileNotFoundError(row["image_path"])
    return rows


def extract(rows, backbone, transform, device, batch_size: int, split: str):
    features, labels, names = [], [], []
    for start in range(0, len(rows), batch_size):
        current = rows[start : start + batch_size]
        tensors = []
        for row in current:
            with Image.open(str(row["image_path"])) as image:
                tensors.append(transform(image.convert("RGB")))
        batch = torch.stack(tensors).to(device)
        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = backbone(batch)
            else:
                output = backbone(batch)
        if output.ndim != 2:
            raise RuntimeError(f"DINOv3 output must be [batch, feature], got {tuple(output.shape)}")
        features.append(torch.nn.functional.normalize(output.float(), dim=1).cpu().numpy())
        labels.extend(int(row["label"]) for row in current)
        names.extend(str(row["image_name"]) for row in current)
        print(f"extract {split}: {min(start + batch_size, len(rows))}/{len(rows)}")
    return np.concatenate(features), np.asarray(labels, dtype=np.int64), names


def score_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = scores >= threshold
    tp = int(np.sum((labels == 1) & pred))
    fp = int(np.sum((labels == 0) & pred))
    fn = int(np.sum((labels == 1) & ~pred))
    tn = int(np.sum((labels == 0) & ~pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": float(threshold), "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a clean DINOv3 image classifier head.")
    parser.add_argument("--train-manifest", type=Path, default=Path(r"C:\fire_detection_clean\data\image_manifests_official_v1\train.csv"))
    parser.add_argument("--val-manifest", type=Path, default=Path(r"C:\fire_detection_clean\data\image_manifests_official_v1\calib.csv"))
    parser.add_argument("--model", default="hf-hub:timm/vit_base_patch16_dinov3.lvd1689m")
    parser.add_argument("--cache-dir", type=Path, default=Path(r"C:\AI\hf_cache\hub"))
    parser.add_argument("--output-dir", type=Path, default=Path(r"C:\fire_detection_clean\runs\dinov3_official_v1"))
    parser.add_argument("--extract-batch-size", type=int, default=8)
    parser.add_argument("--head-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(args.cache_dir.parent)
    os.environ["HF_HUB_CACHE"] = str(args.cache_dir)
    train_rows, val_rows = load_rows(args.train_manifest), load_rows(args.val_manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")
        torch.set_float32_matmul_precision("high")
    import timm
    from timm.data import create_transform, resolve_model_data_config
    print(f"loading frozen backbone: {args.model}")
    backbone = timm.create_model(args.model, pretrained=True, num_classes=0).eval().to(device)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    transform = create_transform(**resolve_model_data_config(backbone), is_training=False)
    train_x, train_y, _ = extract(train_rows, backbone, transform, device, args.extract_batch_size, "train")
    val_x, val_y, val_names = extract(val_rows, backbone, transform, device, args.extract_batch_size, "calib")
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    dimension = int(train_x.shape[1])
    counts = np.bincount(train_y, minlength=2).astype(np.float64)
    weights = torch.tensor(len(train_y) / (2.0 * np.maximum(counts, 1.0)), dtype=torch.float32, device=device)
    head = LinearHead(dimension, args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)), batch_size=args.head_batch_size, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(args.seed))
    best_loss, stale = float("inf"), 0
    history = []
    checkpoint_path = args.output_dir / "best_head.pt"
    for epoch in range(1, args.epochs + 1):
        head.train()
        total, count = 0.0, 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch_y)
            count += len(batch_y)
        head.eval()
        with torch.inference_mode():
            val_logits = head(torch.from_numpy(val_x).to(device))
            val_loss = criterion(val_logits, torch.from_numpy(val_y).to(device)).item()
            val_scores = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
        metrics = score_metrics(val_y, val_scores, 0.5)
        history.append({"epoch": epoch, "train_loss": total / max(count, 1), "val_loss": val_loss, **{f"{k}_0_5": v for k, v in metrics.items() if k in ("precision", "recall", "f1")}})
        print(f"epoch {epoch:03d} train_loss={total / max(count, 1):.6f} val_loss={val_loss:.6f} P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f}")
        if val_loss < best_loss - 1e-6:
            best_loss, stale = val_loss, 0
            torch.save({"head_state_dict": head.state_dict(), "feature_dimension": dimension, "dropout": args.dropout, "model": args.model, "seed": args.seed, "epoch": epoch}, checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping at epoch {epoch}")
                break
    history_path = args.output_dir / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    head.load_state_dict(checkpoint["head_state_dict"])
    head.eval()
    with torch.inference_mode():
        scores = torch.softmax(head(torch.from_numpy(val_x).to(device)), dim=1)[:, 1].cpu().numpy()
    scan = [score_metrics(val_y, scores, float(t)) for t in np.linspace(0.01, 0.99, 99)]
    best = max(scan, key=lambda row: (row["f1"], row["recall"], row["precision"]))
    score_path = args.output_dir / "calib_scores.csv"
    with score_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_name", "label", "score"])
        writer.writeheader()
        for name, label, score in zip(val_names, val_y, scores):
            writer.writerow({"image_name": name, "label": int(label), "score": float(score)})
    result = {"split": "calib", "model": args.model, "train_manifest": str(args.train_manifest.resolve()), "val_manifest": str(args.val_manifest.resolve()), "checkpoint": str(checkpoint_path.resolve()), "calib_scores": str(score_path.resolve()), "best_calib_metrics": best, "holdout_used": False}
    (args.output_dir / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
