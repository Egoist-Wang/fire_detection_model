"""Train one pure Ultralytics YOLO detector with a frozen clean split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a pure YOLO fire detector.")
    parser.add_argument("--model", type=Path, default=Path(r"C:\fire_detection_clean\yolo26s.pt"))
    parser.add_argument("--data", type=Path, default=Path(r"C:\fire_detection_clean\data\grouped_official_v1\data.yaml"))
    parser.add_argument("--project", type=Path, default=Path(r"C:\fire_detection_clean\runs\pure_yolo"))
    parser.add_argument("--name", default="yolo26s_grouped_v1")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--save-period", type=int, default=-1, help="Save a checkpoint every N epochs; -1 keeps Ultralytics default.")
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--dry-run", action="store_true", help="Run one epoch to validate paths and tensors.")
    args = parser.parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    split_freeze = args.data.parent / "SPLIT_FREEZE.json"
    if not split_freeze.is_file():
        raise FileNotFoundError(split_freeze)

    import yaml

    data_spec = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    train_entries = data_spec.get("train")
    val_entries = data_spec.get("val")
    if train_entries is None or val_entries is None:
        raise ValueError(f"Data YAML must define train and val entries: {args.data}")
    flattened_train_entries = train_entries if isinstance(train_entries, list) else [train_entries]
    holdout_used_for_training = any(
        "holdout" in str(entry).replace("\\", "/").strip("/").lower().split("/")
        for entry in flattened_train_entries
    )

    from ultralytics import YOLO, settings

    settings.update({"sync": False, "hub": False})
    args.project.mkdir(parents=True, exist_ok=True)
    run_name = args.name + ("_smoke" if args.dry_run else "")
    config = {
        "model": str(args.model.resolve()),
        "data": str(args.data.resolve()),
        "split_freeze": str(split_freeze.resolve()),
        "epochs": 1 if args.dry_run else args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "patience": 1 if args.dry_run else args.patience,
        "device": args.device,
        "workers": args.workers,
        "save_period": args.save_period,
        "seed": args.seed,
        "train_data_entries": train_entries,
        "validation_data_entries": val_entries,
        "val_split": "Configured validation entries only",
        "holdout_used_for_training": holdout_used_for_training,
        "historical_weights_used": False,
        "fusion_used": False,
    }
    run_dir = args.project / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "TRAIN_CONFIG.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    model = YOLO(str(args.model))
    model.train(
        data=str(args.data),
        epochs=config["epochs"],
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        patience=config["patience"],
        seed=args.seed,
        deterministic=True,
        optimizer="auto",
        project=str(args.project),
        name=run_name,
        exist_ok=False,
        pretrained=True,
        val=True,
        plots=True,
        save=True,
        save_period=args.save_period,
        amp=True,
        close_mosaic=10,
    )
    print(f"run_dir={args.project / run_name}")


if __name__ == "__main__":
    main()
