"""Build a leakage-resistant YOLO dataset from the untouched raw export.

The script deliberately ignores every historical train/val directory and every
historical model artifact. Content-near-duplicate clusters are assigned as
whole groups, so images within one dHash cluster cannot cross a split.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FAMILY_RE = re.compile(r"^(.*?)[_-](\d{2,})$")
D_HASH_DISTANCE = 8
BOUNDARY_DISTANCE = 12
SPLIT_RATIOS = {"train": 0.65, "calib": 0.15, "holdout": 0.20}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def family_of(name: str) -> str:
    match = FAMILY_RE.match(Path(name).stem)
    return match.group(1) if match else Path(name).stem


def dhash64(path: Path) -> int:
    with Image.open(path) as image:
        gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    value = 0
    for row in range(8):
        start = row * 9
        for col in range(8):
            value = (value << 1) | int(pixels[start + col] > pixels[start + col + 1])
    return value


def coco_label_lines(annotations: list[dict], image: dict, cat_to_cls: dict[int, int]) -> list[str]:
    width = float(image["width"])
    height = float(image["height"])
    lines: list[str] = []
    for ann in annotations:
        x, y, box_w, box_h = (float(value) for value in ann["bbox"])
        if box_w <= 0 or box_h <= 0:
            raise ValueError(f"non-positive bbox in {image['file_name']}: {ann['bbox']}")
        x = max(0.0, min(x, width))
        y = max(0.0, min(y, height))
        box_w = max(0.0, min(box_w, width - x))
        box_h = max(0.0, min(box_h, height - y))
        if box_w == 0 or box_h == 0:
            continue
        cx = (x + box_w / 2.0) / width
        cy = (y + box_h / 2.0) / height
        lines.append(
            f"{cat_to_cls[int(ann['category_id'])]} {cx:.6f} {cy:.6f} "
            f"{box_w / width:.6f} {box_h / height:.6f}"
        )
    return lines


def link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != source.stat().st_size:
            raise RuntimeError(f"existing output differs in size: {target}")
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a clean grouped YOLO dataset.")
    parser.add_argument("--raw-root", type=Path, default=Path(r"C:\fire_detection_clean\official_raw"))
    parser.add_argument("--output", type=Path, default=Path(r"C:\fire_detection_clean\data\grouped_official_v1"))
    parser.add_argument("--seed-tag", default="20261001_official_raw_v1")
    args = parser.parse_args()

    image_dir = args.raw_root / "images"
    coco_path = args.raw_root / "train_coco.json"
    image_label_path = args.raw_root / "train_image.json"
    for required in (image_dir, coco_path, image_label_path):
        if not required.exists():
            raise FileNotFoundError(required)

    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    images = {
        str(row["file_name"]): row
        for row in coco["images"]
        if Path(str(row["file_name"])).suffix.lower() in IMAGE_EXTS
        and "(1)" not in Path(str(row["file_name"])).stem
    }
    if len(images) != 1100:
        raise RuntimeError(f"expected 1100 official images after duplicate exclusion, got {len(images)}")
    image_labels = json.loads(image_label_path.read_text(encoding="utf-8"))
    if set(images) != set(image_labels):
        raise RuntimeError("COCO and image-level label universes differ")

    annotations: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        annotations[int(ann["image_id"])].append(ann)
    categories = sorted(coco["categories"], key=lambda row: int(row["id"]))
    cat_to_cls = {int(row["id"]): index for index, row in enumerate(categories)}

    # Build content clusters without reading labels, then assign whole clusters.
    hashes = {name: dhash64(image_dir / name) for name in sorted(images)}
    names = sorted(images)
    parent = {name: name for name in names}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            first, second = sorted((root_left, root_right))
            parent[second] = first

    near_edges = 0
    for index, left in enumerate(names):
        left_hash = hashes[left]
        for right in names[index + 1 :]:
            if (left_hash ^ hashes[right]).bit_count() <= D_HASH_DISTANCE:
                near_edges += 1
                union(left, right)

    clusters: dict[str, list[str]] = defaultdict(list)
    for name in names:
        clusters[find(name)].append(name)
    cluster_list = sorted(clusters.values(), key=lambda members: min(members))
    seed = int(hashlib.sha256(args.seed_tag.encode("utf-8")).hexdigest()[:16], 16)
    random.Random(seed).shuffle(cluster_list)
    targets = {split: ratio * len(names) for split, ratio in SPLIT_RATIOS.items()}
    assigned: dict[str, list[str]] = {split: [] for split in SPLIT_RATIOS}
    for cluster in cluster_list:
        split = min(
            SPLIT_RATIOS,
            key=lambda candidate: (
                max(0, len(assigned[candidate]) + len(cluster) - targets[candidate])
                / targets[candidate],
                len(assigned[candidate]) / targets[candidate],
                candidate,
            ),
        )
        assigned[split].extend(cluster)

    initial_membership = {name: split for split, members in assigned.items() for name in members}
    boundary_endpoints: set[str] = set()
    cross_distance_counts: dict[str, int] = defaultdict(int)
    for index, left in enumerate(names):
        left_split = initial_membership[left]
        for right in names[index + 1 :]:
            if initial_membership[right] == left_split:
                continue
            distance = (hashes[left] ^ hashes[right]).bit_count()
            if distance <= BOUNDARY_DISTANCE:
                cross_distance_counts[str(distance)] += 1
                boundary_endpoints.update((left, right))

    membership = {
        name: split for name, split in initial_membership.items() if name not in boundary_endpoints
    }
    if not membership or not all(any(split == candidate for split in membership.values()) for candidate in SPLIT_RATIOS):
        raise RuntimeError("boundary quarantine emptied a split; revise the pre-registered grouping protocol")

    old_split_dirs = {
        "train": args.output / "train",
        "calib": args.output / "calib",
        "holdout": args.output / "holdout",
    }
    for split_dir in old_split_dirs.values():
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "labels").mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    link_modes: dict[str, int] = defaultdict(int)
    seen_hashes: dict[str, str] = {}
    split_names: dict[str, list[str]] = defaultdict(list)
    for name, image in sorted(images.items()):
        family = family_of(name)
        if name not in membership:
            continue
        split = membership[name]
        source = image_dir / name
        if not source.exists():
            raise FileNotFoundError(source)
        image_hash = sha256(source)
        if image_hash in seen_hashes:
            raise RuntimeError(f"byte duplicate in official universe: {name} and {seen_hashes[image_hash]}")
        seen_hashes[image_hash] = name
        target_image = args.output / split / "images" / name
        link_modes[link_or_copy(source, target_image)] += 1

        target_label = args.output / split / "labels" / f"{Path(name).stem}.txt"
        lines = coco_label_lines(annotations[int(image["id"])], image, cat_to_cls)
        target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        split_names[split].append(name)
        raw_label = image_labels[name]
        label = int(raw_label if isinstance(raw_label, (int, bool)) else raw_label.get("label", raw_label.get("category")))
        rows.append({
            "image_name": name,
            "family": family,
            "split": split,
            "image_level_label": label,
            "source_path": str(source.resolve()),
            "image_sha256": image_hash,
            "label_path": str(target_label.resolve()),
            "box_count": len(lines),
        })

    for split, names in split_names.items():
        image_list = args.output / split / f"{split}.txt"
        image_list.write_text(
            "\n".join(str((args.output / split / "images" / name).resolve()) for name in sorted(names)) + "\n",
            encoding="utf-8",
        )

    quarantined_rows = []
    for name in sorted(boundary_endpoints):
        raw_label = image_labels[name]
        label = int(raw_label if isinstance(raw_label, (int, bool)) else raw_label.get("label", raw_label.get("category")))
        quarantined_rows.append({
            "image_name": name,
            "family": family_of(name),
            "initial_split": initial_membership[name],
            "image_level_label": label,
            "image_sha256": sha256(image_dir / name),
            "dhash64": f"{hashes[name]:016x}",
        })
    if quarantined_rows:
        with (args.output / "quarantine_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(quarantined_rows[0]))
            writer.writeheader()
            writer.writerows(quarantined_rows)

    yaml_text = (
        f"path: {args.output.resolve().as_posix()}\n"
        f"train: train/train.txt\n"
        f"val: calib/calib.txt\n"
        "nc: 1\n"
        "names:\n"
        "  0: fire\n"
    )
    (args.output / "data.yaml").write_text(yaml_text, encoding="utf-8")
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "protocol": "raw_dhash_cluster_grouped_v2",
        "seed_tag": args.seed_tag,
        "raw_root": str(args.raw_root.resolve()),
        "dhash_distance": D_HASH_DISTANCE,
        "boundary_quarantine_distance": BOUNDARY_DISTANCE,
        "dhash_edges_le8": near_edges,
        "content_clusters": len(cluster_list),
        "largest_cluster_sizes": sorted((len(group) for group in cluster_list), reverse=True)[:20],
        "initial_target_ratios": SPLIT_RATIOS,
        "initial_counts": {split: len(items) for split, items in sorted(assigned.items())},
        "cross_split_boundary_edges": dict(sorted(cross_distance_counts.items(), key=lambda item: int(item[0]))),
        "quarantine_count": len(boundary_endpoints),
        "quarantine_names": sorted(boundary_endpoints),
        "counts": {split: len(names) for split, names in sorted(split_names.items())},
        "positive_counts": {
            split: sum(int(row["image_level_label"]) for row in rows if row["split"] == split)
            for split in sorted(split_names)
        },
        "box_counts": {
            split: sum(int(row["box_count"]) for row in rows if row["split"] == split)
            for split in sorted(split_names)
        },
        "exact_sha_cross_split": False,
        "dhash_cross_split_le8_after_quarantine": 0,
        "dhash_cross_split_le12_after_quarantine": 0,
        "link_modes": dict(link_modes),
        "historical_split_inputs_used": False,
        "historical_weights_used": False,
    }
    (args.output / "SPLIT_FREEZE.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
