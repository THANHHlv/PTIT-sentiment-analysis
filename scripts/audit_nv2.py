"""Tạo mẫu rà tách từ/slang và cặp gần trùng từ train; không sửa dữ liệu."""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from ptit_sentiment.data.annotation_review import near_pairs, redact_preview
from ptit_sentiment.data.validate import read_csv
from ptit_sentiment.preprocessing.normalize import normalize_text
from ptit_sentiment.preprocessing.tokenize import tokenize_vietnamese

AMBIGUOUS = re.compile(r"(?iu)\b(?:k|dc|đc|mn|sv)\b")
NEGATION = re.compile(r"(?iu)\b(?:không|chưa|chẳng|chả|đừng|ko|khum|hok)\b")
EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF]")
CODE = re.compile(r"(?iu)\b(?=[a-z\d]*\d)[a-z]{1,8}\d{2,}[a-z\d]*\b")


def audit(train_csv, output_dir, sample_size=150, seed=42):
    """Xuất CSV cần người rà và JSON thống kê; chỉ dùng train CSV làm đầu vào."""
    frame = read_csv(train_csv)
    if not 1 <= sample_size <= len(frame):
        raise ValueError("sample_size phải nằm trong khoảng 1..số dòng train.")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Thư mục output đã có dữ liệu; chọn tên mới.")
    output.mkdir(parents=True, exist_ok=True)
    labels = frame["label"].unique()
    base = sample_size // len(labels)
    required = set()
    for pattern in (EMOJI, NEGATION, AMBIGUOUS, CODE):
        candidates = frame[frame["text"].str.contains(pattern)].drop(index=list(required), errors="ignore")
        required.update(candidates.sample(n=min(5, len(candidates)), random_state=seed).index)
    if len(required) > sample_size:
        raise ValueError("sample_size quá nhỏ để chứa các nhóm kiểm toán bắt buộc.")
    chosen = frame.loc[sorted(required)].copy()
    for label, part in frame.groupby("label"):
        need = max(0, base - int(chosen["label"].eq(label).sum()))
        available = part.drop(index=chosen.index, errors="ignore")
        if need:
            chosen = pd.concat([chosen, available.sample(n=min(need, len(available)), random_state=seed)])
    remainder = sample_size - len(chosen)
    if remainder:
        chosen = pd.concat([chosen, frame.drop(chosen.index).sample(n=remainder, random_state=seed)])
    rows = []
    for row in chosen.itertuples(index=False):
        original = row.text
        normalized = normalize_text(original)
        rows.append({
            "id": row.id, "post_id": row.post_id, "label_ai": row.label,
            "text_preview": redact_preview(original),
            "normalized": normalized,
            "segmented": tokenize_vietnamese(normalized),
            "ambiguous_slang": ",".join(sorted(set(AMBIGUOUS.findall(original.lower())))),
            "has_negation": bool(NEGATION.search(original)),
            "has_emoji": bool(EMOJI.search(original)),
            "has_code_candidate": bool(CODE.search(original)),
            "segmentation_issue": "", "label_review": "", "reviewer_note": "",
        })
    pd.DataFrame(rows).to_csv(output / "tokenization_review.csv", index=False, encoding="utf-8-sig")
    pairs = near_pairs(frame[["id", "text"]].to_dict("records"))
    labels_by_id = frame.set_index("id")["label"].to_dict()
    pair_rows = [{**pair, "label_a": labels_by_id[pair["id_a"]],
                  "label_b": labels_by_id[pair["id_b"]], "review_decision": ""}
                 for pair in pairs]
    pd.DataFrame(pair_rows, columns=["id_a", "id_b", "similarity", "label_a", "label_b", "review_decision"]).to_csv(
        output / "near_duplicate_review.csv", index=False, encoding="utf-8-sig")
    summary = {
        "input": str(train_csv), "sample_rows": len(rows),
        "sample_ai_label_counts": pd.Series([row["label_ai"] for row in rows]).value_counts().to_dict(),
        "ambiguous_slang_rows": sum(bool(row["ambiguous_slang"]) for row in rows),
        "emoji_rows": sum(row["has_emoji"] for row in rows),
        "negation_rows": sum(row["has_negation"] for row in rows),
        "code_candidate_rows": sum(row["has_code_candidate"] for row in rows),
        "near_duplicate_pairs_train": len(pair_rows),
        "near_duplicate_pairs_label_disagreement": sum(p["label_a"] != p["label_b"] for p in pair_rows),
        "note": "Mẫu dùng nhãn AI và preview có che một số thông tin; cần người rà thủ công. Không thay đổi nhãn hoặc CSV nguồn.",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", default="data/splits/train.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(audit(args.train_csv, args.output_dir, args.sample_size, args.seed))
