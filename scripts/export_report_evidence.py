"""Lưu bằng chứng nhỏ cho báo cáo; không sao chép text bình luận."""
import argparse
from math import isclose
from pathlib import Path
import shutil

import pandas as pd

from ptit_sentiment.common import file_hash, read_json, run_cli, write_json
from ptit_sentiment.data.split import load_split_set
from ptit_sentiment.evaluation.metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root, output = Path(args.run_dir), Path(args.output_dir)
    if output.exists():
        raise ValueError("Thư mục bằng chứng đã có; chọn tên mới.")
    frames, manifest, manifest_hash = load_split_set(args.splits_dir)
    if manifest["provenance"] != "real":
        raise ValueError("Bằng chứng báo cáo phải lấy từ dữ liệu thật, không phải fixture.")
    verification = read_json(root / "verification.json")
    if verification["split_manifest_sha256"] != manifest_hash:
        raise ValueError("Bằng chứng kiểm tra không thuộc manifest hiện hành.")
    paths = sorted((root / "evaluation").glob("*/metrics.json"))
    if not paths:
        raise ValueError("Chưa có metrics.json.")
    evidence = {"run": str(root), "provenance": manifest["provenance"],
                "dataset_version": manifest["dataset_version"],
                "labeling_policy": manifest["labeling_policy"],
                "label_source_counts": manifest["label_source_counts"],
                "annotation_status_counts": manifest["annotation_status_counts"],
                "human_labels_confirmed": manifest["human_labels_confirmed"],
                "test_previously_used": verification["test_previously_used"],
                "limitation": manifest["limitation"],
                "source_sha256": manifest["source_sha256"], "manifest_sha256": manifest_hash,
                "splits": {name: {key: info[key] for key in ("rows", "posts", "label_counts", "sha256")}
                           for name, info in manifest["splits"].items()}, "models": []}
    summaries, results = [], {}
    table = pd.read_csv(root / "comparison.csv")
    majority = frames["train"]["label"].value_counts().idxmax()
    for path in paths:
        result = read_json(path)
        if (result["split_manifest_sha256"] != manifest_hash or result["split"] != "test"
                or result["evaluation_data_sha256"] != manifest["splits"]["test"]["sha256"]
                or result["provenance"] != manifest["provenance"]
                or result["n_samples"] != len(frames["test"])):
            raise ValueError("Kết quả không thuộc test của manifest đã chọn.")
        name = result["model_name"]
        if name in results:
            raise ValueError(f"Trùng tên mô hình: {name}")
        predictions = pd.read_csv(path.parent / "predictions.csv", dtype=str, keep_default_na=False)
        for column, source in (("id", "id"), ("text", "text"), ("true_label", "label")):
            if predictions[column].tolist() != frames["test"][source].tolist():
                raise ValueError(f"Dự đoán không khớp thứ tự/nội dung test: {name}")
        recomputed = compute_metrics(predictions["true_label"], predictions["predicted_label"])
        if any(result[key] != value for key, value in recomputed.items()):
            raise ValueError(f"Metrics không khớp dự đoán: {name}")
        compared = table.loc[table.model_name == name]
        if len(compared) != 1 or any(not isclose(float(compared.iloc[0][key]), result[key], abs_tol=1e-12)
                                     for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1")):
            raise ValueError(f"Bảng so sánh không khớp metrics: {name}")
        info = result["training_metadata"]
        if info["split_manifest_sha256"] != manifest_hash or info["fit_split"] != "train" or info["refit_train_validation"]:
            raise ValueError(f"Metadata huấn luyện không hợp lệ: {name}")
        score = info.get("validation_macro_f1")
        if name == "majority_baseline":
            score = compute_metrics(frames["validation"]["label"], [majority] * len(frames["validation"]))["macro_f1"]
        if score is None:
            raise ValueError(f"Thiếu điểm lựa chọn validation: {name}")
        summaries.append({"model_name": name, "validation_macro_f1": score,
                          "parameters": info.get("selected_parameters", {"strategy": "most_frequent"}),
                          "device": info["device"], "fit_split": info["fit_split"],
                          "refit_train_validation": info["refit_train_validation"],
                          "split_manifest_sha256": manifest_hash,
                          "source": str(path), "source_sha256": file_hash(path)})
        results[name] = result
        evidence["models"].append({
            "source": str(path), "sha256": file_hash(path),
            **{key: result[key] for key in ("model_name", "accuracy", "macro_precision",
                                           "macro_recall", "macro_f1", "n_samples",
                                           "per_label", "confusion_matrix")}})
    if set(table.model_name) != set(results) or len(table) != len(results):
        raise ValueError("Danh sách mô hình trong bảng so sánh không khớp kết quả.")
    output.mkdir(parents=True)
    write_json(output / "evidence_real.json", evidence)
    for name in ("comparison.csv", "comparison.png"):
        shutil.copyfile(root / name, output / name)
    write_json(output / "training_summary.json", summaries)
    shutil.copyfile(root / "verification.json", output / "verification.json")
    write_json(output / "phobert_status.json", {
        "status": "not_trained_on_current_split", "split_manifest_sha256": manifest_hash,
        "note": "Checkpoint smoke/bộ chia cũ đã lưu trữ, không phải kết quả của bộ hiện hành.",
        "historical_archive": "outputs/archives/project_cleanup_20261023.zip"})
    (output / "README.md").write_text(
        "# Bằng chứng báo cáo hiện hành\n\n"
        f"Dataset: {manifest['dataset_version']}; nguồn: {root.as_posix()}.\n\n"
        "evidence_real.json chứa số mẫu, nguồn nhãn, hash, độ đo và ma trận; "
        "training_summary.json chứa lựa chọn validation của toàn bộ mô hình. "
        "comparison.csv/.png đối chiếu cùng test. verification.json là biên bản lần chạy gốc.\n\n"
        "Nhãn AI chưa thay nhãn người độc lập; test đã dùng trước đó. "
        "PhoBERT chưa huấn luyện trên bộ chia hiện hành; xem phobert_status.json. "
        "Không có điểm smoke hoặc dữ liệu tổng hợp trong bảng kết quả này.\n",
        encoding="utf-8")
    print(f"Đã lưu bằng chứng {manifest['provenance']}: {output}")


if __name__ == "__main__":
    run_cli(main)
