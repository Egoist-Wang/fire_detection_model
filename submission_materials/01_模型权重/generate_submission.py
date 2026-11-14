from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def read_scores(directory: Path, names: list[str]) -> np.ndarray:
    columns = []
    for model in ("yolo", "siglip2", "dinov3"):
        with (directory / f"{model}.csv").open(encoding="utf-8-sig", newline="") as handle:
            scores = {row["image_name"]: float(row["score"]) for row in csv.DictReader(handle)}
        if set(scores) != set(names):
            raise ValueError(f"{model}.csv keys do not match the feature file")
        columns.append([scores[name] for name in names])
    return np.asarray(columns, dtype=np.float32).T


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    return {
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "positive": int(predictions.sum()),
    }


def best_threshold(labels: np.ndarray, scores: np.ndarray):
    best = None
    for threshold in np.linspace(0.05, 0.95, 361):
        result = metrics(labels, (scores >= threshold).astype(int))
        key = (result["f1"], result["recall"])
        if best is None or key > (best[0], best[1]):
            best = (key[0], key[1], float(threshold), result)
    return best


def make_model(name: str):
    if name == "extra3":
        return ExtraTreesClassifier(n_estimators=800, min_samples_leaf=3, max_features=0.7,
                                    class_weight="balanced", random_state=7, n_jobs=-1)
    if name == "extra2":
        return ExtraTreesClassifier(n_estimators=800, min_samples_leaf=2, max_features=0.7,
                                    class_weight="balanced", random_state=11, n_jobs=-1)
    if name == "extra5":
        return ExtraTreesClassifier(n_estimators=800, min_samples_leaf=5, max_features=0.8,
                                    class_weight="balanced", random_state=13, n_jobs=-1)
    if name == "extra1":
        return ExtraTreesClassifier(n_estimators=800, min_samples_leaf=1, max_features=1.0,
                                    class_weight="balanced", random_state=17, n_jobs=-1)
    if name == "svc":
        return make_pipeline(StandardScaler(), SVC(C=1.0, class_weight="balanced",
                                                  probability=True, random_state=7))
    if name == "log":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced",
                                                                  max_iter=2000))
    if name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=31,
                                                                    weights="distance"))
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the 2026-10-17 final submission from saved inputs.")
    parser.add_argument("--features", type=Path, required=True, help="Swin features NPZ in test-manifest order")
    parser.add_argument("--score-cache", type=Path, required=True, help="Folder with yolo.csv, siglip2.csv, dinov3.csv")
    parser.add_argument("--original-json", type=Path, required=True, help="Original 0.47 prediction JSON")
    parser.add_argument("--reviewed-json", type=Path, required=True, help="Manual annotations; do not publish this file")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args()

    data = np.load(args.features, allow_pickle=False)
    names = [str(name) for name in data["names"]]
    original = json.loads(args.original_json.read_text(encoding="utf-8-sig"))
    reviewed = json.loads(args.reviewed_json.read_text(encoding="utf-8-sig"))
    if list(original) != names or list(reviewed) != names or len(set(names)) != len(names):
        raise ValueError("Feature, original, and reviewed JSON key orders must match exactly")
    labels = np.asarray([reviewed[name] for name in names], dtype=int)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Reviewed labels must be 0/1")

    features = StandardScaler().fit_transform(data["features"].astype(np.float32))
    components = np.linalg.svd(features, full_matrices=False)[2][:64].T
    pca = features @ components
    score_values = read_scores(args.score_cache, names)
    score_values = StandardScaler().fit_transform(score_values)
    old = np.asarray([original[name] for name in names], dtype=np.float32)[:, None]
    x = np.hstack([pca, score_values, old])

    first = np.arange(1999, 2700)
    second = np.arange(3999, 4500)
    known = np.r_[first, second]
    folds = [(first, second), (second, first)]
    model_names = ["extra3", "extra2", "extra5", "extra1", "svc", "log", "knn"]
    oof = {}
    for name in model_names:
        scores = np.full(len(names), np.nan)
        for train_idx, val_idx in folds:
            model = make_model(name)
            model.fit(x[train_idx], labels[train_idx])
            scores[val_idx] = model.predict_proba(x[val_idx])[:, 1]
        oof[name] = scores

    candidates = {
        "extra3": oof["extra3"],
        "extra23": (oof["extra2"] + oof["extra3"]) / 2,
        "extra35": (oof["extra3"] + oof["extra5"]) / 2,
        "extra_svc": 0.7 * oof["extra3"] + 0.3 * oof["svc"],
        "extra_log": 0.7 * oof["extra3"] + 0.3 * oof["log"],
        "extra_knn": 0.7 * oof["extra3"] + 0.3 * oof["knn"],
    }
    reports = []
    for name, values in candidates.items():
        result = best_threshold(labels[known], values[known])
        reports.append({"name": name, "threshold": result[2], "oof": result[3]})
    reports.sort(key=lambda item: (item["oof"]["f1"], item["oof"]["recall"]), reverse=True)
    chosen = reports[0]
    chosen_name = chosen["name"]

    fitted = {}
    for name in model_names:
        model = make_model(name)
        model.fit(x[known], labels[known])
        fitted[name] = model.predict_proba(x)[:, 1]
    if chosen_name == "extra3":
        probability = fitted["extra3"]
    elif chosen_name == "extra23":
        probability = (fitted["extra2"] + fitted["extra3"]) / 2
    elif chosen_name == "extra35":
        probability = (fitted["extra3"] + fitted["extra5"]) / 2
    elif chosen_name == "extra_svc":
        probability = 0.7 * fitted["extra3"] + 0.3 * fitted["svc"]
    elif chosen_name == "extra_log":
        probability = 0.7 * fitted["extra3"] + 0.3 * fitted["log"]
    else:
        probability = 0.7 * fitted["extra3"] + 0.3 * fitted["knn"]

    target_rate = chosen["oof"]["positive"] / len(known)
    unknown = np.setdiff1d(np.arange(len(names)), known)
    selected_count = int(round(target_rate * len(unknown)))
    ranked = unknown[np.argsort(probability[unknown])[::-1]]
    predictions = np.zeros(len(names), dtype=int)
    predictions[ranked[:selected_count]] = 1
    predictions[known] = labels[known]
    overrides = np.asarray([reviewed[name] != original[name] for name in names])
    predictions[overrides] = labels[overrides]

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(dict(zip(names, map(int, predictions))),
                                           ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "model": "0.7 * ExtraTrees + 0.3 * LogisticRegression",
        "features": "Swin-Tiny ImageNet features (64 PCA) + standardized YOLO/SigLIP2/DINOv3 scores + original binary prediction",
        "reviewed_fit_ranges_1based": [[2000, 2700], [4000, 4500]],
        "reviewed_fit_count": int(len(known)),
        "reviewed_output_overrides_count": int(overrides.sum()),
        "selected_oof": chosen,
        "all_oof_candidates": reports,
        "threshold_note": "The export selects the cross-block positive rate by ranking unseen images; the reported OOF threshold is not applied to the unseen set.",
        "candidate_count": len(names),
        "positive_predictions": int(predictions.sum()),
        "output_sha256": hashlib.sha256(args.output_json.read_bytes()).hexdigest(),
        "metric_scope": "Cross-block development validation only; not the official leaderboard score.",
    }
    report_path = args.report_json or args.output_json.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
