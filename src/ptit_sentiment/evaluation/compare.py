"""Tổng hợp metrics.json của mô hình đã đánh giá trên cùng tập dữ liệu."""
import argparse
from pathlib import Path

import pandas as pd

from ptit_sentiment.common import read_json, run_cli, write_json

COLUMNS = ("model_name", "split", "provenance", "n_samples",
           "accuracy", "macro_precision", "macro_recall", "macro_f1")


def compare_results(inputs, output):
    """Chỉ lấy metric có thực, không trộn bộ chia/tập test hay lần chạy trùng tên."""
    rows, signature, names = [], None, set()
    for path in inputs:
        result = read_json(path)
        current = tuple(result[key] for key in
                        ("split_manifest_sha256", "evaluation_data_sha256", "split", "provenance", "n_samples"))
        if signature is not None and current != signature:
            raise ValueError("Không thể so sánh các kết quả từ bộ chia/tập dữ liệu khác nhau.")
        signature = current
        if result["model_name"] in names:
            raise ValueError(f"Trùng model_name {result['model_name']}; chọn một lần chạy cho mỗi mô hình.")
        names.add(result["model_name"])
        rows.append({key: result[key] for key in COLUMNS})
    if not rows:
        raise ValueError("Không có metrics.json; phải chạy evaluate trước.")
    output = Path(output)
    if any(output.with_suffix(ext).exists() for ext in (".csv", ".json", ".png")):
        raise ValueError("Bảng so sánh đã tồn tại; chọn --output mới.")
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows).sort_values("macro_f1", ascending=False, kind="stable")
    table.to_csv(output, index=False, encoding="utf-8")
    write_json(output.with_suffix(".json"), {"evaluation_signature": list(signature),
                                            "models": table.to_dict(orient="records"),
                                            "sources": [str(path) for path in inputs]})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    axis = table.set_index("model_name")[["accuracy", "macro_f1"]].plot.bar(
        figsize=(9, 5), ylim=(0, 1.1), rot=20)
    axis.set_ylabel("Điểm")
    axis.set_xlabel("Mô hình")
    axis.set_title(f"So sánh trên {signature[2]} — {signature[3]}")
    axis.figure.tight_layout()
    axis.figure.savefig(output.with_suffix(".png"), dpi=160)
    plt.close(axis.figure)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="Các metrics.json thực tế")
    parser.add_argument("--output", required=True, help="CSV bảng so sánh")
    args = parser.parse_args()
    print(compare_results(args.inputs, args.output).to_string(index=False))


if __name__ == "__main__":
    run_cli(main)
