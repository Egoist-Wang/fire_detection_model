# Fire Detection Model

Clean retraining project for image-level fire classification from the official dataset.

The current candidate combines a YOLO26m detector and a frozen DINOv3 image classifier.
The decision rule is:

```text
YOLO fire confidence >= 0.02 OR DINOv3 fire score >= 0.62
```

The rule reached F1 0.9536 and recall 1.0 on the 137-image calibration split. The
calibration split was used for rule selection; the competition test set must not be used
to tune thresholds.

## Contents

- `weights/`: DINOv3/SigLIP2 heads; the YOLO checkpoint is attached to the GitHub Release
- `inference/`: standalone prediction entry point
- `src/`: training and evaluation scripts
- `configs/`: relative-path model and threshold configuration
- `results/`: calibration result record
- `docs/`: data protocol and reproducibility notes

The official images and annotations are intentionally excluded from this repository.

## Inference

Install dependencies and ensure the DINOv3 backbone is available through the official
Hugging Face/timm identifier in `configs/final_candidate.json`.

```powershell
python inference/predict_fusion.py `
  --images C:\path\to\test_images `
  --output outputs\predictions.csv `
  --cache-dir C:\AI\hf_cache\hub
```

The output contains `image_name`, `yolo_score`, `dino_score`, and the final binary
prediction.

## Competition submission artifact

The latest reviewed competition candidate is stored at
`submissions/SUBMIT_20261017_EXTRA_ORIG_RATE.json`. Its method and validation record
are documented in `docs/submission_20261017_extra_orig_rate.md`. This file contains
predictions in test-manifest order; it does not contain the official test images or
labels.

## Third-party components

This repository uses Ultralytics YOLO, PyTorch, timm, and the official DINOv3 model.
Their licenses and notices remain applicable. This repository contains the project
configuration, inference code, and trained checkpoints.

## Model files

- `weights/dinov3_head.pt` and `weights/siglip2_head.pt` are stored in this repository.
- `weights/yolo26m_epoch77.pt` (166.97 MB) is distributed as a GitHub Release asset because it exceeds the 100 MB normal-file limit.
- See `MODEL_CARD.md` for checkpoint metadata, SHA256 hashes, and usage notes.
- Release: https://github.com/Egoist-Wang/fire_detection_model/releases/tag/v1.0.0

The duplicate 166.97 MB checkpoint under `submission_materials/01_模型权重/` is intentionally omitted from the repository; use `weights/` or the Release asset instead.