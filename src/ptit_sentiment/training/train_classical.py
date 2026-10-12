"""Huấn luyện bốn mô hình; chọn tham số bằng Macro-F1 validation."""
import argparse
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from time import perf_counter
import warnings

import joblib
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score

from ptit_sentiment.common import environment_versions, file_hash, load_config, run_cli, write_json
from ptit_sentiment.data.split import load_split_set
from ptit_sentiment.labels import LABELS, LABEL_TO_ID
from ptit_sentiment.models.classical import MODEL_NAMES, fit_baseline, make_pipeline
from ptit_sentiment.preprocessing.tokenize import TextPreprocessor


def fit_classifier(classifier, matrix, labels, retry_max_iter=100000):
    """Fit sparse train features; retry a nonconverged SVM once, then fail clearly."""
    attempts = []
    for attempt in range(2):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(matrix, labels)
        convergence = [str(item.message) for item in caught
                       if issubclass(item.category, ConvergenceWarning)]
        for item in caught:
            if not issubclass(item.category, ConvergenceWarning):
                warnings.warn(item.message, item.category)
        attempts.append({"max_iter": getattr(classifier, "max_iter", None),
                         "n_iter": int(classifier.n_iter_) if hasattr(classifier, "n_iter_") else None,
                         "convergence_warnings": convergence})
        if not convergence:
            return attempts
        if attempt == 0 and hasattr(classifier, "max_iter") and retry_max_iter > classifier.max_iter:
            classifier.set_params(max_iter=retry_max_iter)
        else:
            break
    raise ValueError(f"SVM chưa hội tụ; không lưu mô hình không hợp lệ: {attempts}")


def train_models(splits_dir, config_path, artifact_dir, output_dir):
    """Fit chỉ train; chọn pipeline bằng validation, lưu mà không refit trên test."""
    frames, manifest, split_hash = load_split_set(splits_dir)
    started = datetime.now(timezone.utc).isoformat()
    run_clock = perf_counter()
    config = load_config(config_path)
    artifact_dir, output_dir = Path(artifact_dir), Path(output_dir)
    names = (*MODEL_NAMES, "majority_baseline")
    if any((artifact_dir / f"{name}.joblib").exists() or
           (artifact_dir / f"{name}.metadata.json").exists() for name in names):
        raise ValueError("Artifact đã có; chọn --artifact-dir mới để bảo toàn lần chạy trước.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Thư mục log train đã có dữ liệu; chọn --output-dir mới.")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    train, validation = frames["train"], frames["validation"]
    base_metadata = {
        "kind": "classical", "label_to_id": LABEL_TO_ID,
        "split_manifest_sha256": split_hash,
        "split_file_hashes": {name: info["sha256"] for name, info in manifest["splits"].items()},
        "provenance": manifest["provenance"], "config": config,
        "versions": environment_versions(), "selection_metric": "validation_macro_f1",
        "fit_split": "train", "refit_train_validation": False,
        "dataset_version": manifest.get("dataset_version"),
        "source_sha256": manifest.get("source_sha256"),
        "labeling_policy": manifest.get("labeling_policy"),
        "label_source_counts": manifest.get("label_source_counts"),
        "annotation_status_counts": manifest.get("annotation_status_counts"),
        "human_labels_confirmed": manifest.get("human_labels_confirmed", False),
        "config_sha256": file_hash(config_path), "started_at_utc": started,
        "sample_counts": {name: len(frame) for name, frame in frames.items()},
        "tie_break": "first candidate in YAML order; strict greater validation Macro-F1",
        "device": "cpu",
    }
    # Stateless NV2 processing is identical for every candidate: segment once.
    processor = TextPreprocessor(**config.get("preprocessing", {}))
    preprocess_clock = perf_counter()
    train_tokens = processor.fit_transform(train["text"].tolist())
    validation_tokens = processor.transform(validation["text"].tolist())
    base_metadata["preprocessing_seconds"] = perf_counter() - preprocess_clock
    summaries = []
    for name in MODEL_NAMES:
        feature, algorithm = name.split("_")
        parameters = config["grid"]["nb_alpha" if algorithm == "nb" else "svm_c"]
        candidates = list(product(config["grid"]["ngram_range"], parameters))
        if not candidates:
            raise ValueError(f"Lưới tham số {name} rỗng.")
        best_model, best_score, best_params, trials = None, -1.0, None, []
        for ngram, parameter in candidates:
            trial_clock = perf_counter()
            model = make_pipeline(feature, algorithm, config, ngram, parameter)
            model.steps[0] = ("preprocess", processor)
            vectorizer, classifier = model.named_steps["vectorizer"], model.named_steps["classifier"]
            try:
                train_matrix = vectorizer.fit_transform(train_tokens)
                validation_matrix = vectorizer.transform(validation_tokens)
                fit_clock = perf_counter()
                attempts = fit_classifier(classifier, train_matrix, train["label"].tolist(),
                                          config.get("svm", {}).get("retry_max_iter", 100000))
                fit_seconds = perf_counter() - fit_clock
            except ValueError as error:
                raise ValueError(f"{name}: không fit được vocabulary/classifier; kiểm tra dữ liệu và min_df/max_df: {error}") from error
            predicted = classifier.predict(validation_matrix)
            score = float(f1_score(validation["label"], predicted, labels=list(LABELS),
                                   average="macro", zero_division=0))
            params = {"ngram_range": list(ngram), "alpha" if algorithm == "nb" else "C": parameter}
            trials.append({"parameters": params, "validation_macro_f1": score,
                           "fit_seconds": fit_seconds, "total_seconds": perf_counter() - trial_clock,
                           "train_shape": list(train_matrix.shape), "train_nnz": train_matrix.nnz,
                           "fit_attempts": attempts})
            # Nếu hòa, lấy cấu hình xuất hiện trước trong YAML để tái lập.
            if score > best_score:
                best_model, best_score, best_params = model, score, params
        artifact = artifact_dir / f"{name}.joblib"
        joblib.dump(best_model, artifact)
        reloaded = joblib.load(artifact)
        if not (reloaded.predict(validation["text"].tolist()) == best_model.predict(validation["text"].tolist())).all():
            raise ValueError(f"{name}: dự đoán thay đổi sau khi lưu/tải joblib.")
        metadata = {**base_metadata, "model_name": name, "baseline": False,
                    "selected_parameters": best_params, "validation_macro_f1": best_score,
                    "search_seconds": sum(t["total_seconds"] for t in trials),
                    "classifier_parameters": best_model.named_steps["classifier"].get_params(),
                    "artifact_sha256": file_hash(artifact), "reload_validation_equal": True,
                    "completed_at_utc": datetime.now(timezone.utc).isoformat()}
        write_json(artifact.with_suffix(".metadata.json"), metadata)
        write_json(output_dir / f"{name}_validation_search.json", {"model_name": name, "trials": trials, **metadata})
        summaries.append({"model_name": name, "validation_macro_f1": best_score, "parameters": best_params})
    baseline_clock = perf_counter()
    baseline = fit_baseline(train["text"].tolist(), train["label"].tolist())
    artifact = artifact_dir / "majority_baseline.joblib"
    joblib.dump(baseline, artifact)
    write_json(artifact.with_suffix(".metadata.json"), {
        **base_metadata, "model_name": "majority_baseline", "baseline": True,
        "majority_label": str(baseline.classes_[baseline.class_prior_.argmax()]),
        "artifact_sha256": file_hash(artifact), "fit_seconds": perf_counter() - baseline_clock,
    })
    write_json(output_dir / "training_summary.json", summaries)
    write_json(output_dir / "selection_lock.json", {
        "selected_models": summaries, "split_manifest_sha256": split_hash,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_predictions_computed": False, "total_seconds": perf_counter() - run_clock,
        "artifacts": {name: file_hash(artifact_dir / f"{name}.joblib") for name in names},
    })
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--config", default="configs/classical.yaml")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    for result in train_models(args.splits_dir, args.config, args.artifact_dir, args.output_dir):
        print(result)


if __name__ == "__main__":
    run_cli(main)
