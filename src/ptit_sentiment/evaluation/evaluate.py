"""Đánh giá artifact; lệnh riêng, không huấn luyện/chọn tham số."""
import argparse
from pathlib import Path

from ptit_sentiment.common import file_hash, run_cli
from ptit_sentiment.data.split import load_split_set
from ptit_sentiment.evaluation.metrics import export_evaluation
from ptit_sentiment.predict import load_artifact, predict_with_artifact


def evaluate_model(model_path, splits_dir, output_dir, split="test", segmenter_dir=None):
    """Từ chối đánh giá trên bộ chia khác bộ dùng khi huấn luyện."""
    frames, manifest, split_hash = load_split_set(splits_dir)
    model, metadata = load_artifact(model_path, segmenter_dir)
    if metadata["split_manifest_sha256"] != split_hash:
        raise ValueError("Bộ chia đánh giá khác bộ chia lúc train. Dùng chung manifest cho mọi mô hình.")
    frame = frames[split]
    predictions = predict_with_artifact(model, metadata, frame)
    return export_evaluation(output_dir, predictions, {
        "model_name": metadata["model_name"], "split": split,
        "provenance": manifest["provenance"], "split_manifest_sha256": split_hash,
        "evaluation_data_sha256": file_hash(Path(splits_dir) / f"{split}.csv"),
        "artifact": str(Path(model_path).resolve()),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--segmenter-dir")
    args = parser.parse_args()
    metrics = evaluate_model(args.model, args.splits_dir, args.output_dir, args.split, args.segmenter_dir)
    print({key: metrics[key] for key in ("model_name", "split", "accuracy", "macro_f1")})


if __name__ == "__main__":
    run_cli(main)
