# Competition submission: 2026-10-17

Recommended file: `submissions/SUBMIT_20261017_EXTRA_ORIG_RATE.json`

The candidate combines frozen Swin-Tiny image features, YOLO, SigLIP2, and DINOv3 scores, plus the binary result from the original 0.47 submission. An ExtraTrees classifier was fitted using the manually reviewed image ranges 2000-2700 and 4000-4500. The reviewed labels were preserved in the exported submission.

Cross-block validation on the two reviewed ranges:

- F1: 0.66445
- Precision: 0.62500
- Recall: 0.70922

Submission checks:

- 5,313 image keys
- Key order matches the test manifest
- Values are integer `0` or `1`
- Positive predictions: 688
- SHA256: `5d9426f0013c88ad63e96cc5f12a7ac078d32d64f1edcf07881b3d09b3b7f8`

The JSON is a competition prediction artifact. The official images, labels, caches, and local model feature files remain excluded from this repository.
