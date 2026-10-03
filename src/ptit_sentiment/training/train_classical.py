"""Huấn luyện bốn mô hình; chọn tham số bằng Macro-F1 validation."""
import argparse
from itertools import product
from pathlib import Path

import joblib
from sklearn.metrics import f1_score

from ptit_sentiment.common import environment_versions, load_config, run_cli, write_json
from ptit_sentiment.data.split import load_split_set
from ptit_sentiment.labels import LABELS, LABEL_TO_ID
from ptit_sentiment.models.classical import MODEL_NAMES, fit_baseline, make_pipeline


def train_models(splits_dir, config_path, artifact_dir, output_dir):
    """Fit chỉ train; chọn pipeline bằng validation, lưu mà không refit trên test."""
    frames, manifest, split_hash = load_split_set(splits_dir)
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
    }
    summaries = []
    for name in MODEL_NAMES:
        feature, algorithm = name.split("_")
        parameters = config["grid"]["nb_alpha" if algorithm == "nb" else "svm_c"]
        candidates = list(product(config["grid"]["ngram_range"], parameters))
        if not candidates:
            raise ValueError(f"Lưới tham số {name} rỗng.")
        best_model, best_score, best_params, trials = None, -1.0, None, []
        for ngram, parameter in candidates:
            model = make_pipeline(feature, algorithm, config, ngram, parameter)
            try:
                model.fit(train["text"].tolist(), train["label"].tolist())
            except ValueError as error:
                raise ValueError(f"{name}: không fit được vocabulary/classifier; kiểm tra dữ liệu và min_df/max_df: {error}") from error
            predicted = model.predict(validation["text"].tolist())
            score = float(f1_score(validation["label"], predicted, labels=list(LABELS),
                                   average="macro", zero_division=0))
            params = {"ngram_range": list(ngram), "alpha" if algorithm == "nb" else "C": parameter}
            trials.append({"parameters": params, "validation_macro_f1": score})
            # Nếu hòa, lấy cấu hình xuất hiện trước trong YAML để tái lập.
            if score > best_score:
                best_model, best_score, best_params = model, score, params
        artifact = artifact_dir / f"{name}.joblib"
        joblib.dump(best_model, artifact)
        reloaded = joblib.load(artifact)
        if not (reloaded.predict(validation["text"].tolist()) == best_model.predict(validation["text"].tolist())).all():
            raise ValueError(f"{name}: dự đoán thay đổi sau khi lưu/tải joblib.")
        metadata = {**base_metadata, "model_name": name, "baseline": False,
                    "selected_parameters": best_params, "validation_macro_f1": best_score}
        write_json(artifact.with_suffix(".metadata.json"), metadata)
        write_json(output_dir / f"{name}_validation_search.json", {"model_name": name, "trials": trials, **metadata})
        summaries.append({"model_name": name, "validation_macro_f1": best_score, "parameters": best_params})
    baseline = fit_baseline(train["text"].tolist(), train["label"].tolist())
    artifact = artifact_dir / "majority_baseline.joblib"
    joblib.dump(baseline, artifact)
    write_json(artifact.with_suffix(".metadata.json"), {
        **base_metadata, "model_name": "majority_baseline", "baseline": True,
        "majority_label": str(baseline.classes_[baseline.class_prior_.argmax()]),
    })
    write_json(output_dir / "training_summary.json", summaries)
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
