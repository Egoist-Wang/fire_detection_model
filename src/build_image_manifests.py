"""Create image-level classifier manifests from the frozen clean split."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean image-level CSV manifests.")
    parser.add_argument(
        "--data-root", type=Path,
        default=Path(r"C:\fire_detection_clean\data\grouped_official_v1"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(r"C:\fire_detection_clean\data\image_manifests_official_v1"),
    )
    args = parser.parse_args()

    rows = list(csv.DictReader((args.data_root / "manifest.csv").open("r", newline="", encoding="utf-8")))
    if not rows:
        raise RuntimeError("The clean manifest is empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "calib", "holdout"):
        selected = [row for row in rows if row["split"] == split]
        if not selected:
            raise RuntimeError(f"No rows for split {split}")
        output = args.output_dir / f"{split}.csv"
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image_name", "image_path", "label"])
            writer.writeheader()
            for row in sorted(selected, key=lambda item: item["image_name"]):
                writer.writerow({
                    "image_name": row["image_name"],
                    "image_path": row["source_path"],
                    "label": int(row["image_level_label"]),
                })
        print(f"{split}: {len(selected)} -> {output}")


if __name__ == "__main__":
    main()
