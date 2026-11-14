# Data Protocol

The clean dataset was rebuilt from the official raw export and official annotations.
Content-similar image families were grouped with dHash distance 8. Boundary samples
with cross-split dHash distance up to 12 were quarantined.

Final development counts:

- train: 626 images
- calibration: 137 images
- internal holdout: 184 images
- quarantine: 153 images

The current final-fit YOLO checkpoint was trained on train plus the former internal
holdout (810 images), while calibration remained the validation split. The quarantine
set was excluded. No historical fire weights, historical splits, old embeddings, or
old fusion outputs were used as training inputs.
