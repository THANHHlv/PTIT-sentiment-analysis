"""Chạy CLI thực tế bằng dữ liệu tổng hợp; không gọi fine-tune PhoBERT."""
import argparse
from datetime import datetime
from pathlib import Path
import os
import subprocess
import sys

import joblib
import pandas as pd

from ptit_sentiment.common import write_json, read_json
from ptit_sentiment.data.split import check_disjoint, load_split_set
from ptit_sentiment.models.classical import MODEL_NAMES
from ptit_sentiment.preprocessing.normalize import normalize_text
from ptit_sentiment.preprocessing.tokenize import TextPreprocessor


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="smoke_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    if not args.run_name.replace("_", "").replace("-", "").isalnum():
        raise SystemExit("run-name chỉ nhận chữ, số, _ và -.")
    splits = Path("data/splits") / args.run_name
    artifacts = Path("artifacts") / args.run_name
    outputs = Path("outputs") / args.run_name
    if any(path.exists() for path in (splits, artifacts, outputs)):
        raise SystemExit("Tên lần chạy đã tồn tại; chọn --run-name mới.")
    outputs.mkdir(parents=True)
    commands = []

    def run(module, *options):
        command = [sys.executable, "-m", module, *map(str, options)]
        print("Chạy:", " ".join(command), flush=True)
        completed = subprocess.run(command, text=True, encoding="utf-8", capture_output=True,
                                   env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        commands.append({"command": command, "exit_code": completed.returncode,
                         "stdout": completed.stdout, "stderr": completed.stderr})
        write_json(outputs / "commands.json", commands)
        if completed.returncode:
            print(completed.stdout, completed.stderr)
            raise SystemExit(completed.returncode)
        print(completed.stdout[:800], flush=True)

    run("ptit_sentiment.data.validate", "--input", "tests/fixtures/synthetic.csv",
        "--output", outputs / "data_validation.json")
    run("ptit_sentiment.data.split", "--input", "tests/fixtures/synthetic.csv",
        "--output-dir", splits, "--provenance", "synthetic")
    run("ptit_sentiment.training.train_classical", "--splits-dir", splits,
        "--artifact-dir", artifacts, "--output-dir", outputs / "training")
    metrics = []
    for name in (*MODEL_NAMES, "majority_baseline"):
        run("ptit_sentiment.evaluation.evaluate", "--model", artifacts / f"{name}.joblib",
            "--splits-dir", splits, "--output-dir", outputs / "evaluation" / name)
        metrics.append(outputs / "evaluation" / name / "metrics.json")
    run("ptit_sentiment.evaluation.compare", "--inputs", *metrics, "--output", outputs / "comparison.csv")
    run("ptit_sentiment.predict", "--model", artifacts / "bow_nb.joblib",
        "--text", "Thầy hỗ trợ rất nhiệt tình, mình hài lòng.", "--output", outputs / "new_sentence.csv")
    run("ptit_sentiment.predict", "--model", artifacts / "tfidf_svm.joblib",
        "--input", "tests/fixtures/new_comments.csv", "--output", outputs / "new_comments.csv")
    frames, manifest, _ = load_split_set(splits)
    check_disjoint(frames)
    for name in MODEL_NAMES:
        model = joblib.load(artifacts / f"{name}.joblib")
        reloaded = joblib.load(artifacts / f"{name}.joblib")
        texts = frames["test"]["text"].tolist()
        assert (model.predict(texts) == reloaded.predict(texts)).all()
        predictions = pd.read_csv(outputs / "evaluation" / name / "predictions.csv", dtype=str, keep_default_na=False)
        assert predictions["text"].tolist() == frames["test"]["text"].tolist()
        assert predictions["id"].tolist() == frames["test"]["id"].tolist()
        assert bool([col for col in predictions if col.startswith("probability_")]) == name.endswith("_nb")
        preprocessed = model.named_steps["preprocess"].transform(frames["train"]["text"].tolist())
        vectorizer = model.named_steps["vectorizer"]
        analyzer = vectorizer.build_analyzer()
        train_features = {feature for text in preprocessed for feature in analyzer(text)}
        assert set(vectorizer.vocabulary_) <= train_features
        evaluated = read_json(outputs / "evaluation" / name / "metrics.json")
        report = pd.read_csv(outputs / "evaluation" / name / "classification_report.csv", index_col="label")
        assert list(report.index) == ["positive", "neutral", "negative", "macro_avg"]
        assert report.loc["macro_avg", "f1"] == evaluated["macro_f1"]
        assert sum(map(sum, evaluated["confusion_matrix"]["values"])) == len(texts)
        assert sum(evaluated["confusion_matrix"]["values"][i][i] for i in range(3)) / len(texts) == evaluated["accuracy"]
        if name.endswith("_svm"):
            assert not hasattr(model, "predict_proba")
    from PIL import Image
    with Image.open(outputs / "comparison.png") as chart:
        assert chart.width > 100 and chart.height > 100
    assert len(pd.read_csv(outputs / "comparison.csv")) == 5
    example = "  SV ko  thích lịch học 😢 https://example.org  "
    write_json(outputs / "preprocessing_example.json", {
        "raw": example, "normalized": normalize_text(example),
        "tokenized": TextPreprocessor().fit_transform([example])[0]})
    write_json(outputs / "verification.json", {
        "status": "passed", "provenance": "synthetic",
        "no_post_id_id_text_overlap": True, "joblib_reload_matches": True,
        "raw_text_preserved": True, "vocabulary_from_train_only": True,
        "metric_exports_and_comparison_chart_checked": True, "svm_no_probability": True,
        "models": [*MODEL_NAMES, "majority_baseline"], "phobert_finetune": "not_run",
        "paths": {"splits": str(splits), "artifacts": str(artifacts), "outputs": str(outputs)},
        "split_statistics": {name: {key: info[key] for key in ("rows", "posts", "label_counts")}
                             for name, info in manifest["splits"].items()},
    })
    print("Smoke thành công; chỉ kiểm tra code trên dữ liệu tổng hợp:", outputs)


if __name__ == "__main__":
    main()
