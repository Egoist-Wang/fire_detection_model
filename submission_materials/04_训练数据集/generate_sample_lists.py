from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_names(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [row["image_name"] for row in csv.DictReader(handle)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create name-only manifests; labels are never written.")
    parser.add_argument("--official-manifests", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--reviewed-json", type=Path, required=True)
    parser.add_argument("--original-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    official_rows = []
    uses = {"train": "official classifier training; also YOLO training",
            "holdout": "added to final YOLO training only",
            "calib": "validation/calibration for official models"}
    for split, usage in uses.items():
        for name in read_names(args.official_manifests / f"{split}.csv"):
            official_rows.append({"image_name": name, "split": split, "recorded_use": usage})
    with (args.output_dir / "official_training_samples.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_name", "split", "recorded_use"])
        writer.writeheader(); writer.writerows(official_rows)

    names = read_names(args.test_manifest)
    reviewed = json.loads(args.reviewed_json.read_text(encoding="utf-8-sig"))
    original = json.loads(args.original_json.read_text(encoding="utf-8-sig"))
    if list(reviewed) != names or list(original) != names:
        raise ValueError("Test manifest and JSON key order do not match")
    fit_indexes = set(range(1999, 2700)) | set(range(3999, 4500))
    rows = []
    for i, name in enumerate(names):
        if i in fit_indexes:
            rows.append({"manifest_index_1based": i + 1, "image_name": name,
                         "recorded_use": "manual label used to fit/cross-validate final fusion"})
        elif reviewed[name] != original[name]:
            rows.append({"manifest_index_1based": i + 1, "image_name": name,
                         "recorded_use": "FINAL-vs-original output override; label not included"})
    with (args.output_dir / "manually_reviewed_test_sample_names.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["manifest_index_1based", "image_name", "recorded_use"])
        writer.writeheader(); writer.writerows(rows)
    print(f"official sample rows: {len(official_rows)}")
    print(f"manual/override name rows: {len(rows)} (labels omitted)")


if __name__ == "__main__":
    main()
