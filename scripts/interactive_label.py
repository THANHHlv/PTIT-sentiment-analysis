"""Công cụ gán nhãn tương tác CLI cho 4 thành viên.

Giúp người gán nhãn thao tác nhanh trên Terminal mà không cần sửa file CSV bằng Excel
(tránh lỗi mã hóa UTF-8 BOM, mất định dạng dòng, hoặc sửa nhầm text/post_id).

Quy tắc AGENTS.md:
- Mỗi bình luận nhận đúng 1 nhãn: positive, neutral, negative.
- suggested_label chỉ là gợi ý, người gán nhãn tự quyết định nhãn thật và xác nhận 'yes'.
- Không được sửa id, post_id, text, role.

Cách dùng:
    $python = ".\\.venv\\Scripts\\python.exe"
    &$python scripts/interactive_label.py --annotator 1
"""
import argparse
import sys
from pathlib import Path
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")

from ptit_sentiment.common import run_cli
from ptit_sentiment.labels import LABELS

LABEL_MAP = {
    "1": "positive",
    "pos": "positive",
    "+": "positive",
    "2": "neutral",
    "neu": "neutral",
    "0": "neutral",
    "3": "negative",
    "neg": "negative",
    "-": "negative",
}


def interactive_session(csv_path, annotator_id):
    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str, encoding="utf-8", keep_default_na=False)
    total = len(df)
    confirmed_count = (df["confirmed"].str.strip().str.lower() == "yes").sum()

    print("=" * 65)
    print(f"   HỆ THỐNG GÁN NHÃN CẢM XÚC PTIT — NGƯỜI GÁN NHÃN {annotator_id}")
    print(f"   File: {csv_path}")
    print(f"   Tiến độ: {confirmed_count}/{total} bình luận đã xác nhận ({confirmed_count/total*100:.1f}%)")
    print("=" * 65)
    print("Phím tắt:")
    print("  [1] positive  [2] neutral  [3] negative")
    print("  [a] Chấp nhận gợi ý (suggested_label)")
    print("  [s] Bỏ qua câu này (skip sang câu tiếp theo)")
    print("  [b] Quay lại câu trước")
    print("  [n] Thêm/sửa ghi chú (notes)")
    print("  [q] Lưu và thoát")
    print("-" * 65)

    idx = 0
    # Tìm câu đầu tiên chưa confirmed
    for i in range(total):
        if df.at[i, "confirmed"].strip().lower() != "yes":
            idx = i
            break

    while 0 <= idx < total:
        row = df.iloc[idx]
        is_confirmed = row["confirmed"].strip().lower() == "yes"
        status_tag = f"[ĐÃ XÁC NHẬN: {row['label']}]" if is_confirmed else "[CHƯA GÁN]"

        print(f"\n--- [{idx + 1}/{total}] {status_tag} (Vai trò: {row['role']}) ---")
        print(f"Post ID: {row['post_id']} | Comment ID: {row['id']}")
        print(f"Nội dung:")
        for line in row["text"].split("\n"):
            print(f"    │ {line}")
        print(f"Gợi ý AI : {row.get('suggested_label', '') or '(chưa có)'}")
        if row.get("notes", "").strip():
            print(f"Ghi chú   : {row['notes']}")

        prompt = "Lựa chọn [1/2/3/a/s/b/n/q]: "
        try:
            choice = input(prompt).strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nĐang lưu và thoát...")
            choice = "q"

        if choice == "q":
            df.to_csv(csv_path, index=False, encoding="utf-8")
            print(f"Đã lưu {csv_path.name}.")
            break

        elif choice in LABEL_MAP:
            selected_label = LABEL_MAP[choice]
            df.at[idx, "label"] = selected_label
            df.at[idx, "confirmed"] = "yes"
            print(f"-> Đã gán: {selected_label} (confirmed: yes)")
            df.to_csv(csv_path, index=False, encoding="utf-8")
            idx += 1

        elif choice == "a":
            sugg = row.get("suggested_label", "").strip()
            if sugg in LABELS:
                df.at[idx, "label"] = sugg
                df.at[idx, "confirmed"] = "yes"
                print(f"-> Đã nhận gợi ý: {sugg} (confirmed: yes)")
                df.to_csv(csv_path, index=False, encoding="utf-8")
                idx += 1
            else:
                print("Không có gợi ý hợp lệ để nhận. Hãy nhập 1, 2 hoặc 3.")

        elif choice == "s":
            print("-> Bỏ qua.")
            idx += 1

        elif choice == "b":
            if idx > 0:
                idx -= 1
            else:
                print("Đã ở câu đầu tiên.")

        elif choice == "n":
            try:
                new_note = input("Nhập ghi chú: ").strip()
                df.at[idx, "notes"] = new_note
                df.to_csv(csv_path, index=False, encoding="utf-8")
                print("-> Đã lưu ghi chú.")
            except (KeyboardInterrupt, EOFError):
                pass

        else:
            print("Lệnh không hợp lệ. Vui lòng chọn 1, 2, 3, a, s, b, n hoặc q.")

    confirmed_count = (df["confirmed"].str.strip().str.lower() == "yes").sum()
    print("=" * 65)
    print(f"KẾT THÚC PHIÊN: {confirmed_count}/{total} bình luận đã xác nhận.")
    if confirmed_count == total:
        print("Chúc mừng! Bạn đã hoàn thành toàn bộ phần phân công của mình.")
    else:
        print(f"Còn lại {total - confirmed_count} bình luận cần hoàn thiện.")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotator", type=int, choices=[1, 2, 3, 4], required=True,
                        help="Số thứ tự người gán nhãn (1, 2, 3, hoặc 4)")
    parser.add_argument("--assignment-dir", default="data/labeled/assignments_v1",
                        help="Thư mục phân công")
    args = parser.parse_args()

    csv_path = Path(args.assignment_dir) / f"annotator_{args.annotator}.csv"
    interactive_session(csv_path, args.annotator)


if __name__ == "__main__":
    run_cli(main)
