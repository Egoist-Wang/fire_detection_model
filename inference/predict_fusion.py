"""Run the frozen YOLO26m + DINOv3 image-level fusion candidate."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

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


def image_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extensions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--yolo-weights", type=Path, default=Path("weights/yolo26m_epoch77.pt"))
    parser.add_argument("--dino-head", type=Path, default=Path("weights/dinov3_head.pt"))
    parser.add_argument("--dino-model", default="hf-hub:timm/vit_base_patch16_dinov3.lvd1689m")
    parser.add_argument("--cache-dir", type=Path, default=Path(r"C:\AI\hf_cache\hub"))
    parser.add_argument("--yolo-threshold", type=float, default=0.02)
    parser.add_argument("--dino-threshold", type=float, default=0.62)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    files = image_files(args.images)
    if not files:
        raise RuntimeError(f"No images found: {args.images}")
    for path in (args.yolo_weights, args.dino_head):
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.environ["HF_HOME"] = str(args.cache_dir.parent)
    os.environ["HF_HUB_CACHE"] = str(args.cache_dir)

    from ultralytics import YOLO

    yolo = YOLO(str(args.yolo_weights))
    yolo_scores: dict[str, float] = {}
    predictions = yolo.predict(
        source=[str(path) for path in files],
        imgsz=args.imgsz,
        batch=4,
        conf=0.01,
        iou=0.7,
        device=str(device),
        half=device.type == "cuda",
        stream=True,
        verbose=False,
    )
    for path, result in zip(files, predictions):
        boxes = result.boxes
        values = []
        if boxes is not None and len(boxes):
            values = [
                float(boxes.conf[index].item())
                for index in range(len(boxes))
                if int(boxes.cls[index].item()) == 0
            ]
        yolo_scores[path.name] = max(values, default=0.0)

    import timm
    from timm.data import create_transform, resolve_model_data_config

    backbone = timm.create_model(args.dino_model, pretrained=True, num_classes=0).eval().to(device)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    checkpoint = torch.load(args.dino_head, map_location=device, weights_only=False)
    head = LinearHead(int(checkpoint["feature_dimension"]), float(checkpoint["dropout"])).to(device)
    head.load_state_dict(checkpoint["head_state_dict"])
    head.eval()
    transform = create_transform(**resolve_model_data_config(backbone), is_training=False)
    dino_scores: dict[str, float] = {}
    for start in range(0, len(files), args.batch_size):
        current = files[start : start + args.batch_size]
        tensors = []
        for path in current:
            with Image.open(path) as image:
                tensors.append(transform(image.convert("RGB")))
        batch = torch.stack(tensors).to(device)
        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features = backbone(batch)
            else:
                features = backbone(batch)
            features = torch.nn.functional.normalize(features.float(), dim=1)
            scores = torch.softmax(head(features), dim=1)[:, 1].cpu().tolist()
        dino_scores.update({path.name: float(score) for path, score in zip(current, scores)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_name", "yolo_score", "dino_score", "prediction"])
        writer.writeheader()
        for path in files:
            yolo_score = yolo_scores[path.name]
            dino_score = dino_scores[path.name]
            prediction = int(yolo_score >= args.yolo_threshold or dino_score >= args.dino_threshold)
            writer.writerow({"image_name": path.name, "yolo_score": yolo_score, "dino_score": dino_score, "prediction": prediction})
    print(f"wrote {len(files)} predictions to {args.output}")


if __name__ == "__main__":
    main()
