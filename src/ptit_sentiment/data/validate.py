"""Kiểm tra CSV thật hoặc tổng hợp mà không sửa nội dung gốc."""
import argparse
import unicodedata
from pathlib import Path

import pandas as pd

from ptit_sentiment.common import run_cli, write_json
from ptit_sentiment.labels import LABELS

REQUIRED_COLUMNS = ("id", "post_id", "text", "label")


def text_key(text):
    """Khóa so trùng: NFC, casefold và gộp khoảng trắng; không sửa text."""
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def read_csv(path, labeled=True):
    """Đọc mọi cột dạng chuỗi (giữ số 0 đầu id), kiểm tra hợp đồng."""
    frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    validate_frame(frame, labeled=labeled)
    return frame


def validate_frame(frame, labeled=True):
    """Từ chối thiếu cột, ô rỗng, nhãn sai, id/bình luận trùng."""
    required = REQUIRED_COLUMNS if labeled else ("id", "text")
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Thiếu cột: {sorted(missing)}.")
    if frame.empty:
        raise ValueError("CSV không có bình luận. Hãy nhập dữ liệu vào mẫu.")
    for col in required:
        empty = frame[col].str.strip().eq("")
        if empty.any():
            raise ValueError(f"Cột {col} rỗng tại dòng CSV {(frame.index[empty] + 2).tolist()[:10]}.")
    if frame["id"].str.strip().ne(frame["id"]).any():
        raise ValueError("id không được có khoảng trắng đầu/cuối.")
    if labeled and frame["post_id"].str.strip().ne(frame["post_id"]).any():
        raise ValueError("post_id không được có khoảng trắng đầu/cuối.")
    if labeled:
        unknown = set(frame["label"]) - set(LABELS)
        if unknown:
            raise ValueError(f"Nhãn không hợp lệ: {sorted(unknown)}; chỉ nhận {LABELS}.")
    for name, keys in (("id", frame["id"]), ("bình luận", frame["text"].map(text_key))):
        duplicated = keys.duplicated(keep=False)
        if duplicated.any():
            raise ValueError(f"Trùng {name}; id cần rà soát: {frame.loc[duplicated, 'id'].tolist()[:10]}.")


def statistics(frame):
    """Số bình luận, bài viết và phân bố đủ ba nhãn kể cả support 0."""
    return {
        "rows": len(frame), "posts": int(frame["post_id"].nunique()),
        "label_counts": {label: int(frame["label"].eq(label).sum()) for label in LABELS},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", help="JSON thống kê tùy chọn")
    args = parser.parse_args()
    report = statistics(read_csv(args.input))
    if args.output:
        write_json(args.output, report)
    print(report)


if __name__ == "__main__":
    run_cli(main)
