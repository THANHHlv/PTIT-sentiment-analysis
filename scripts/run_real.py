"""Vận hành bộ chia thật đã xác nhận; train và test là hai lệnh riêng."""
import argparse
from pathlib import Path

from ptit_sentiment.common import load_config, run_cli
from ptit_sentiment.data.split import load_split_set
from ptit_sentiment.training.train_classical import train_models
from ptit_sentiment.evaluation.evaluate import evaluate_model
from ptit_sentiment.evaluation.compare import compare_results

CLASSICAL = ("bow_nb", "tfidf_nb", "bow_svm", "tfidf_svm", "majority_baseline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "evaluate"))
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--classical-config", default="configs/classical.yaml")
    parser.add_argument("--phobert-config", default="configs/phobert_gpu4gb.yaml")
    parser.add_argument("--skip-phobert", action="store_true", help="Chỉ classical, trạng thái chưa đủ sáu mô hình")
    args = parser.parse_args()
    _, manifest, digest = load_split_set(args.splits_dir)
    if manifest.get("provenance") != "real" or not manifest.get("human_labels_confirmed"):
        raise ValueError("Lệnh này chỉ nhận bộ chia thật có xác nhận nhãn; không vận hành dữ liệu minh họa.")
    artifact, output = Path(args.artifact_dir), Path(args.output_dir)
    print(f"Bộ chia thật đã xác nhận: {digest}")
    if args.stage == "train":
        if not args.skip_phobert:
            config = load_config(args.phobert_config)
            if config.get("training", {}).get("max_steps", -1) > 0:
                raise ValueError("Không dùng cấu hình giới hạn smoke max_steps cho thực nghiệm đầy đủ.")
        train_models(args.splits_dir, args.classical_config, artifact / "classical", output / "training/classical")
        if not args.skip_phobert:
            from ptit_sentiment.training.train_phobert import train_phobert
            train_phobert(args.splits_dir, args.phobert_config, artifact / "phobert")
        print("Đã train; chưa đánh giá test. Chốt cấu hình rồi chạy stage evaluate.")
    else:
        models = [(name, artifact / "classical" / f"{name}.joblib") for name in CLASSICAL]
        if not args.skip_phobert:
            models.append(("phobert", artifact / "phobert"))
        missing = [str(path) for _, path in models if not path.exists()]
        if missing:
            raise ValueError(f"Artifact còn thiếu: {missing}. Không tự train trong đánh giá.")
        metrics = []
        for name, path in models:
            destination = output / "evaluation" / name
            evaluate_model(path, args.splits_dir, destination)
            metrics.append(destination / "metrics.json")
        compare_results(metrics, output / "comparison.csv")
        print(f"Kết quả test thật: {output}. Chỉ những mô hình đã đánh giá được đưa vào bảng.")
    if args.skip_phobert:
        print("PhoBERT chưa được thực hiện trong lệnh này; hệ thống chưa đủ yêu cầu sáu mô hình.")


if __name__ == "__main__":
    run_cli(main)