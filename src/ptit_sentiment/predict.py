"""Dự đoán một câu hoặc CSV từ artifact đã lưu, không fit lại."""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from ptit_sentiment.common import read_json, run_cli, write_json
from ptit_sentiment.data.validate import read_csv
from ptit_sentiment.evaluation.metrics import prediction_frame
from ptit_sentiment.labels import LABELS, LABEL_TO_ID
from ptit_sentiment.models.classical import predict_classical


def load_artifact(path, segmenter_dir=None):
    """Đọc artifact trusted local và metadata; PhoBERT import lazy."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact chưa tồn tại: {path}. Chạy train trước hoặc chọn đúng --model.")
    if path.is_dir():
        from ptit_sentiment.models.phobert import PhoBERTPredictor
        predictor = PhoBERTPredictor(path, segmenter_dir)
        return predictor, predictor.metadata
    if path.suffix != ".joblib":
        raise ValueError("Mô hình truyền thống cần tệp .joblib; PhoBERT cần thư mục đã lưu.")
    metadata = read_json(path.with_suffix(".metadata.json"))
    if metadata.get("label_to_id") != LABEL_TO_ID:
        raise ValueError("Ánh xạ nhãn artifact khác labels.py.")
    model = joblib.load(path)
    if set(model.classes_) != set(LABELS):
        raise ValueError("Classifier artifact không có đúng ba nhãn.")
    return model, metadata


def predict_with_artifact(model, metadata, frame):
    """Dự đoán và trả DataFrame thống nhất, giữ nguyên text đầu vào."""
    texts = frame["text"].tolist()
    if metadata["kind"] == "phobert":
        predicted, probabilities, classes = model.predict(texts)
    else:
        predicted, probabilities, classes = predict_classical(model, texts, metadata.get("baseline", False))
    return prediction_frame(frame, predicted, metadata["model_name"], probabilities, classes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Tệp joblib hoặc thư mục PhoBERT local")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text")
    inputs.add_argument("--input", help="CSV UTF-8 ít nhất id,text; label tùy chọn")
    parser.add_argument("--output", help="CSV dự đoán")
    parser.add_argument("--segmenter-dir", help="Đường dẫn VnCoreNLP nếu di chuyển artifact")
    args = parser.parse_args()
    if args.input and not args.output:
        raise ValueError("--input cần --output để lưu kết quả.")
    if args.text is not None:
        if not args.text.strip():
            raise ValueError("Bình luận mới không được rỗng.")
        frame = pd.DataFrame([{"id": "new-1", "text": args.text}])
    else:
        frame = read_csv(args.input, labeled=False)
        if "label" in frame and set(frame["label"]) - set(LABELS):
            raise ValueError("Cột label tùy chọn chứa nhãn không hợp lệ; bỏ cột nếu chưa gán nhãn.")
    if args.output and Path(args.output).exists():
        raise ValueError(f"Tệp {args.output} đã tồn tại; chọn đầu ra mới.")
    model, metadata = load_artifact(args.model, args.segmenter_dir)
    result = predict_with_artifact(model, metadata, frame)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(path, index=False, encoding="utf-8")
        if metadata["kind"] == "phobert":
            write_json(path.with_suffix(".truncation.json"), model.truncation_stats)
        print(f"Đã lưu {len(result)} dự đoán: {path}")
    else:
        print(result.to_json(orient="records", force_ascii=False))


if __name__ == "__main__":
    run_cli(main)
