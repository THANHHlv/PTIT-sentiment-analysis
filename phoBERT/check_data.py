
import pandas as pd
from pathlib import Path

# Xác định thư mục gốc project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Đường dẫn thư mục dữ liệu
DATA_DIR = PROJECT_ROOT / "data" / "splits"

FILES = {
    "train": DATA_DIR / "train.csv",
    "validation": DATA_DIR / "validation.csv",
    "test": DATA_DIR / "test.csv",
}

REQUIRED_COLUMNS = ["text", "label"]

VALID_LABELS = ["positive", "neutral", "negative"]


def check_dataset(name, file_path):
    print(f"\n{'=' * 50}")
    print(f"CHECKING: {name.upper()}")
    print(f"{'=' * 50}")

    if not file_path.exists():
        print(f"Không tìm thấy file: {file_path}")
        return None

    df = pd.read_csv(file_path)

    print("Số lượng dữ liệu:", len(df))
    print("Các cột:", df.columns.tolist())

    # Kiểm tra cột bắt buộc
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        print("Thiếu cột:", missing_columns)
        return None

    # Kiểm tra dữ liệu rỗng
    print("\nSố text bị thiếu:", df["text"].isna().sum())
    print("Số label bị thiếu:", df["label"].isna().sum())

    # Phân bố nhãn
    print("\nPhân bố nhãn:")
    print(df["label"].value_counts(dropna=False))

    # Kiểm tra nhãn không hợp lệ
    invalid_labels = (
        set(df["label"].dropna().unique()) - set(VALID_LABELS)
    )

    print("\nNhãn không hợp lệ:", invalid_labels)

    # Kiểm tra text trùng lặp
    print("Số text trùng lặp:", df["text"].duplicated().sum())

    # Kiểm tra post_id
    if "post_id" in df.columns:
        print("Số bài viết khác nhau:", df["post_id"].nunique())

    return df


if __name__ == "__main__":
    datasets = {}

    for name, path in FILES.items():
        datasets[name] = check_dataset(name, path)
