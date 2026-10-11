"""So sánh thăm dò NV2 trên train/validation hiện tại; không đọc test."""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import f1_score

from ptit_sentiment.common import file_hash
from ptit_sentiment.data.validate import read_csv, text_key
from ptit_sentiment.evaluation.metrics import compute_metrics, error_groups
from ptit_sentiment.labels import LABELS
from ptit_sentiment.models.classical import make_pipeline


def compare(train_csv, validation_csv, config_path, output_dir):
    """Fit từng cấu hình trên train, đo validation và ghi hash/giới hạn rõ ràng."""
    train, validation = read_csv(train_csv), read_csv(validation_csv)
    for column in ("id", "post_id"):
        if set(train[column]) & set(validation[column]):
            raise ValueError(f"Rò rỉ {column} giữa train và validation.")
    if set(train["text"].map(text_key)) & set(validation["text"].map(text_key)):
        raise ValueError("Rò rỉ nội dung giữa train và validation.")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Thư mục output đã có dữ liệu; chọn tên mới.")
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    variants = [
        ("word", "keep", None, 1, False),
        ("word_char", "keep", None, 1, False),
        ("word_char", "keep", "balanced", 1, False),
        ("word_char", "keep_and_map", "balanced", 1, False),
        ("word_char", "keep", "balanced", 2, True),
    ]
    results = []
    for mode, emoji, weight, min_df, sublinear in variants:
        options = {**config["vectorizer"], "min_df": min_df, "sublinear_tf": sublinear}
        model = make_pipeline("tfidf", "svm", config, (1, 2), 1.0,
                              emoji=emoji, class_weight=weight, tfidf_mode=mode,
                              vectorizer_options=options)
        model.fit(train["text"].tolist(), train["label"].tolist())
        predicted = model.predict(validation["text"].tolist())
        score = f1_score(validation["label"], predicted, labels=list(LABELS),
                         average="macro", zero_division=0)
        result = {"tfidf_mode": mode, "emoji": emoji, "class_weight": weight,
                  "min_df": min_df, "sublinear_tf": sublinear,
                  "validation_macro_f1": float(score),
                  "per_label": compute_metrics(validation["label"].tolist(), predicted.tolist())["per_label"]}
        results.append(result)
        frame = validation[["id", "text", "label"]].rename(columns={"label": "true_label"}).copy()
        frame["predicted_label"] = predicted
        error_groups(frame).to_csv(output / f"error_groups_{len(results)}.csv", index=False, encoding="utf-8")
    report = {
        "status": "exploratory_validation_only_manifest_unverified",
        "train_sha256": file_hash(train_csv), "validation_sha256": file_hash(validation_csv),
        "config_sha256": file_hash(config_path), "train_rows": len(train),
        "validation_rows": len(validation), "test_used": False,
        "note": "Không dùng làm kết quả chính thức khi manifest chưa khớp. Không dùng validation để kết luận trên nhãn người; toàn bộ nhãn là AI.",
        "variants": results,
    }
    (output / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", default="data/splits/train.csv")
    parser.add_argument("--validation-csv", default="data/splits/validation.csv")
    parser.add_argument("--config", default="configs/classical.yaml")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    for result in compare(args.train_csv, args.validation_csv, args.config, args.output_dir):
        print(result)
