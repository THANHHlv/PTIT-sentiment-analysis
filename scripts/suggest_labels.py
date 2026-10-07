"""Đề xuất nhãn hỗ trợ người gán nhãn (suggested_label).

Theo AGENTS.md:
"AI hoặc mô hình chỉ được đề xuất suggested_label;
không biến đề xuất thành nhãn chuẩn nếu chưa được người gán nhãn xác nhận."

Script này:
1. Đọc annotator_1.csv .. annotator_4.csv trong thư mục phân công.
2. Dùng mô hình đã có để dự đoán suggested_label cho các bình luận.
3. Cột `label` và `confirmed` vẫn ĐỂ TRỐNG (người gán nhãn tự duyệt và xác nhận).
4. Giữ nguyên id, post_id, text, role.

Cách dùng:
    $python = ".\\.venv\\Scripts\\python.exe"
    &$python scripts/suggest_labels.py \
        --assignment-dir data/labeled/assignments_v1 \
        --model artifacts/complete_20261003/tfidf_svm.joblib
"""
import argparse
import sys
from pathlib import Path
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")

from ptit_sentiment.common import run_cli
from ptit_sentiment.predict import load_artifact, predict_with_artifact


def suggest_for_assignments(assignment_dir, model_path, annotators=(1, 2, 3, 4)):
    directory = Path(assignment_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục: {directory}")

    model, metadata = load_artifact(model_path)
    print(f"Sử dụng mô hình: {metadata.get('model_name', model_path)} ({metadata.get('kind', 'classical')})")

    total_suggested = 0
    for person in annotators:
        csv_path = directory / f"annotator_{person}.csv"
        if not csv_path.exists():
            print(f"Bỏ qua {csv_path} (không tồn tại)")
            continue

        df = pd.read_csv(csv_path, dtype=str, encoding="utf-8", keep_default_na=False)
        needs_suggestion = df["suggested_label"].str.strip().eq("")
        count = int(needs_suggestion.sum())

        if count > 0:
            sub = df[needs_suggestion].copy()
            pred_df = predict_with_artifact(model, metadata, sub)
            df.loc[needs_suggestion, "suggested_label"] = pred_df["predicted_label"].values
            df.to_csv(csv_path, index=False, encoding="utf-8")
            print(f"Người {person} ({csv_path.name}): đã gợi ý {count}/{len(df)} bình luận.")
            total_suggested += count
        else:
            print(f"Người {person} ({csv_path.name}): tất cả {len(df)} bình luận đã có suggested_label.")

    print(f"\nTổng cộng đã đề xuất {total_suggested} nhãn gợi ý.")
    print("LƯU Ý: Cột 'label' và 'confirmed' vẫn để trống để người gán nhãn tự thẩm định.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment-dir", default="data/labeled/assignments_v1",
                        help="Thư mục chứa annotator_{1..4}.csv")
    parser.add_argument("--model", default="artifacts/complete_20261003/tfidf_svm.joblib",
                        help="Đường dẫn mô hình joblib hoặc PhoBERT")
    parser.add_argument("--annotators", nargs="+", type=int, default=[1, 2, 3, 4],
                        help="Danh sách annotator cần chạy (mặc định: 1 2 3 4)")
    args = parser.parse_args()

    suggest_for_assignments(args.assignment_dir, args.model, args.annotators)


if __name__ == "__main__":
    run_cli(main)
