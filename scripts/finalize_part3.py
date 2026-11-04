"""Validate a completed NV3 run, export validation metrics and build a portable handoff."""
import argparse
from datetime import datetime, timezone
import hashlib
from math import isclose
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile

import pandas as pd

from ptit_sentiment.common import file_hash, read_json, run_cli, write_json
from ptit_sentiment.data.split import load_split_set
from ptit_sentiment.evaluation.evaluate import evaluate_model
from ptit_sentiment.evaluation.metrics import compute_metrics
from ptit_sentiment.labels import LABELS
from ptit_sentiment.models.classical import MODEL_NAMES
from ptit_sentiment.predict import load_artifact, predict_with_artifact


def require(condition, message):
    """Integrity checks remain active even if Python is invoked with -O."""
    if not condition:
        raise ValueError(message)


def finalize(run_name, output_dir, archive_path, test_log):
    """Reuse locked models and test outputs; never train, refit, relabel or select on test."""
    require(Path(run_name).name == run_name and run_name not in (".", ".."), "Tên run không hợp lệ.")
    run = Path("outputs") / run_name
    artifacts = Path("artifacts") / run_name
    output, archive = Path(output_dir), Path(archive_path)
    require(not output.exists() and not archive.exists(), "Đầu ra đã tồn tại; chọn tên mới.")
    tests = read_json(test_log)
    require(tests["returncode"] == 0 and re.search(r"Ran \d+ tests", tests["stderr"]),
            "Cần log kiểm thử toàn dự án đã đạt.")
    for path, digest in tests["code_sha256"].items():
        require(file_hash(path) == digest, f"Code đã đổi sau kiểm thử: {path}")
    frames, manifest, split_hash = load_split_set("data/splits")
    require(manifest["provenance"] == "real", "Không dùng fixture cho bàn giao thực nghiệm.")
    require(file_hash("data/labeled/labeled_comments.csv") == manifest["source_sha256"], "Hash labeled sai.")
    lock = read_json(run / "all_selection_lock.json")
    previous_check = read_json(run / "verification.json")
    require(previous_check["all_models_locked_before_test"] and previous_check["data_unchanged"],
            "Thiếu bằng chứng khóa lựa chọn và bảo toàn dữ liệu.")
    require(lock["split_manifest_sha256"] == split_hash and not lock["test_predictions_computed"],
            "Khóa lựa chọn không khớp manifest.")
    require(file_hash(run / "all_selection_lock.json") == previous_check["selection_lock_sha256"],
            "Khóa lựa chọn đã đổi.")
    for path, digest in lock["selection_locks"].items():
        require(file_hash(path) == digest, f"Khóa train đã đổi: {path}")
    names = list(lock["artifacts"])
    require(set((*MODEL_NAMES, "majority_baseline")) <= set(names), "Thiếu tổ hợp bắt buộc/baseline.")
    paths, metadata, test_results = {}, {}, {}
    fixture = pd.read_csv("tests/fixtures/new_comments.csv", dtype=str, keep_default_na=False)
    expected_predictions = {}
    for name in names:
        path = artifacts / ("extended" if name == "tfidf_svm_extended" else "") / (name + ".joblib")
        require(file_hash(path) == lock["artifacts"][name], f"Artifact đã đổi: {name}")
        model, info = load_artifact(path)
        require(info["split_manifest_sha256"] == split_hash and info["fit_split"] == "train"
                and not info["refit_train_validation"], f"Chính sách fit/manifest sai: {name}")
        predictions = pd.read_csv(run / f"evaluation/{name}/predictions.csv", dtype=str, keep_default_na=False)
        for column, source_column in (("id", "id"), ("post_id", "post_id"), ("text", "text"), ("true_label", "label")):
            require(predictions[column].tolist() == frames["test"][source_column].tolist(), f"Thứ tự/cột test sai: {name}/{column}")
        require(set(predictions["model_name"]) == {name}, f"Tên model sai: {name}")
        result = read_json(run / f"evaluation/{name}/metrics.json")
        require(result["split"] == "test" and result["split_manifest_sha256"] == split_hash
                and result["evaluation_data_sha256"] == manifest["splits"]["test"]["sha256"], f"Phiên bản test sai: {name}")
        recomputed = compute_metrics(predictions["true_label"], predictions["predicted_label"])
        require(all(result[key] == value for key, value in recomputed.items()), f"Metrics không khớp predictions: {name}")
        errors = pd.read_csv(run / f"evaluation/{name}/errors.csv", dtype=str, keep_default_na=False)
        require(errors["id"].tolist() == predictions.loc[predictions.true_label != predictions.predicted_label, "id"].tolist(),
                f"Errors không khớp: {name}")
        if name != "majority_baseline":
            feature_check = read_json(run / f"feature_audits/{name}.json")
            require(feature_check["artifact_sha256"] == lock["artifacts"][name]
                    and feature_check["matches_selection_score"], f"Audit đặc trưng sai: {name}")
        paths[name], metadata[name], test_results[name] = path, info, result
        expected_predictions[name] = predict_with_artifact(model, info, fixture)["predicted_label"].tolist()
    examples = pd.read_csv(run / "anonymized_error_examples.csv", dtype=str, keep_default_na=False)
    links = read_json(run / "error_example_links_local.json")
    selected_name = read_json(run / "training/selection_lock.json")["recommended_model"]["model_name"]
    raw_errors = pd.read_csv(run / f"evaluation/{selected_name}/errors.csv", dtype=str, keep_default_na=False).set_index("id")
    prediction_column = "predicted_label" if "predicted_label" in examples else "predicted_label_bow_nb"
    require(len(examples) == len(links) == 25 and set(examples.example_id) == {item["example_id"] for item in links},
            "Ví dụ lỗi không đủ/không khớp đối soát.")
    for item in links:
        raw = raw_errors.loc[item["id"]]
        sample = examples.set_index("example_id").loc[item["example_id"]]
        require(sample["excerpt"] in raw["text"] and sample["true_label_ai"] == raw["true_label"]
                and sample[prediction_column] == raw["predicted_label"], "Trích đoạn lỗi không khớp dữ liệu thật.")
        require(hashlib.sha256(raw["text"].encode("utf-8")).hexdigest() == item["text_sha256"], "Hash ví dụ lỗi sai.")
    output.mkdir(parents=True)
    validation_results = {}
    for name in names:
        value = evaluate_model(paths[name], "data/splits", output / "validation" / name, split="validation")
        if name != "majority_baseline":
            require(isclose(value["macro_f1"], metadata[name]["validation_macro_f1"], rel_tol=1e-12, abs_tol=1e-12),
                    f"Điểm validation thay đổi: {name}")
        validation_results[name] = value
    learned = [name for name in names if name != "majority_baseline"]
    primary = max(learned, key=lambda name: validation_results[name]["macro_f1"])
    rows = [{"model_name": name, "mandatory": name in MODEL_NAMES,
             "validation_macro_f1": validation_results[name]["macro_f1"],
             **{key: test_results[name][key] for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1")}}
            for name in names]
    pd.DataFrame(rows).to_csv(output / "results.csv", index=False, encoding="utf-8")
    primary_info = {"model_name": primary, "artifact": paths[primary].as_posix(),
                    "artifact_sha256": file_hash(paths[primary]), "selection_metric": "validation_macro_f1",
                    "validation_macro_f1": validation_results[primary]["macro_f1"],
                    "test_used_for_selection": False, "source_selection_lock": (run / "all_selection_lock.json").as_posix(),
                    "note": "Summarizes the existing validation choice; does not retrain or change locked artifacts"}
    write_json(output / "primary_model.json", primary_info)
    members = [Path("pyproject.toml"), Path("features.py"), Path(".gitattributes"), *Path("src").rglob("*.py"),
               Path(metadata[primary].get("config_path", "configs/part3.yaml")),
               Path("tests/fixtures/new_comments.csv"), Path("docs/part3_handoff.md"), Path("docs/part3_learning.md"),
               Path("report/report.md"), Path("report/part3_error_analysis.md"),
               Path("scripts/verify_part3_bundle.py"), Path("scripts/run_part3.py"),
               Path("scripts/review_part2.py"), Path("scripts/audit_selected_features.py"),
               Path("scripts/finalize_part3.py"), *Path("tests").glob("*.py"), *Path("tests/fixtures").glob("*.csv"),
               run / "all_selection_lock.json", run / "verification.json", run / "run_context.json",
               run / "part2_audit.json", run / "anonymized_error_examples.csv",
               Path(test_log), output / "results.csv", output / "primary_model.json"]
    for path in paths.values():
        members.extend([path, path.with_suffix(".metadata.json")])
    for folder in (run / "training", run / "training_extended", run / "feature_audits"):
        if folder.exists():
            members.extend(folder.glob("*.json"))
    aggregate_files = ("metrics.json", "metrics.csv", "per_label.csv", "classification_report.csv", "confusion_matrix.csv",
                       "confusion_matrix.png", "error_groups.csv")
    for name in names:
        for directory in (run / "evaluation" / name, output / "validation" / name):
            members.extend(directory / filename for filename in aggregate_files)
    versions = metadata[primary]["versions"]
    requirements = "\n".join(f"{package}=={versions[package]}" for package in
                             ("scikit-learn", "pandas", "numpy", "joblib", "matplotlib", "PyYAML", "underthesea")) + "\n"
    readme = f"""# Bàn giao NV3 — mô hình truyền thống

Gói dùng riêng trong nhóm; chứa artifact và kết quả đã kiểm chứng của {run_name}.
Mô hình chính: {primary}, chọn bằng Macro-F1 validation, không chọn bằng test.
Đọc docs/part3_handoff.md và report/report.md.
Dữ liệu thật nhận riêng từ người 1, cùng dataset {manifest['dataset_version']}.
Gói không chứa CSV bình luận thật, predictions/errors có text gốc hoặc ID bài Facebook.
25 ví dụ lỗi đã ẩn danh có trong báo cáo. Artifact giữ metadata/hash nguồn.

## Windows PowerShell (chạy trong thư mục giải nén)

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements-part3.txt
.\\.venv\\Scripts\\python.exe -m pip install -e . --no-deps
.\\.venv\\Scripts\\python.exe scripts/verify_part3_bundle.py --root .
.\\.venv\\Scripts\\python.exe -m ptit_sentiment.predict --model {paths[primary].as_posix()} --text "Mình chưa hài lòng với lịch học."
```

Verifier kiểm tra hash và tải {len(names)} mô hình dự đoán fixture, không cần dữ liệu thật/không train.
Muốn train/evaluate lại, nhận data/ hiện hành từ người 1 và dùng run-name mới.
Nhãn hiện tại đều AI, test đã dùng trước đó: chưa phải đánh giá độc lập bằng nhãn người.
PhoBERT thuộc người 4, chưa được fine-tune trong gói này.
"""
    # Include figures and linked documents so the report is readable after extraction.
    workspace = Path.cwd().resolve()
    included = set(members)
    pending_documents = [path for path in included if path.suffix == ".md"]
    while pending_documents:
        document = pending_documents.pop()
        for link in re.findall(r"\[[^\]]+\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
            if link.startswith(("https:", "http:", "#")):
                continue
            target = (document.parent / link.split("#")[0]).resolve()
            require(target.is_relative_to(workspace), f"Liên kết ngoài workspace: {document}: {link}")
            require(target.is_file(), f"Thiếu tệp trong báo cáo: {document}: {link}")
            relative = target.relative_to(workspace)
            # Real-data CSVs remain a separate input; do not follow data links into the ZIP.
            if relative.suffix in (".md", ".png", ".yaml", ".py") or relative.name == ".gitattributes":
                if relative not in included:
                    included.add(relative)
                    if relative.suffix == ".md":
                        pending_documents.append(relative)
    payload = {path.as_posix(): path.read_bytes() for path in included}
    payload["requirements-part3.txt"] = requirements.encode("utf-8")
    payload["README_NV3.md"] = readme.encode("utf-8")
    bundle_manifest = {"schema_version": 1, "run_name": run_name, "primary_model": primary,
                       "dataset_version": manifest["dataset_version"], "split_manifest_sha256": split_hash,
                       "label_order": list(LABELS), "human_labels_confirmed": False, "test_previously_used": True,
                       "models": {name: path.as_posix() for name, path in paths.items()},
                       "prediction_check": {"inputs": fixture.to_dict(orient="records"), "expected": expected_predictions},
                       "files_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in sorted(payload.items())},
                       "raw_data_included": False, "raw_predictions_errors_included": False}
    write_json(output / "bundle_manifest.json", bundle_manifest)
    payload["bundle_manifest.json"] = (output / "bundle_manifest.json").read_bytes()
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as package:
        for name, content in sorted(payload.items()):
            package.writestr(name, content)
    # Verify using the extracted source package in an isolated import path, without data/.
    with tempfile.TemporaryDirectory(prefix="ptit_nv3_verify_") as temporary:
        target = Path(temporary)
        with zipfile.ZipFile(archive) as package:
            for name in package.namelist():
                require((target / name).resolve().is_relative_to(target.resolve()), "ZIP có đường dẫn ngoài gói.")
            package.extractall(target)
        code = "import runpy,sys;sys.path.insert(0,'src');sys.argv=['scripts/verify_part3_bundle.py','--root','.'];runpy.run_path('scripts/verify_part3_bundle.py',run_name='__main__')"
        checked = subprocess.run([sys.executable, "-I", "-B", "-c", code], cwd=target,
                                 capture_output=True, text=True, encoding="utf-8", errors="replace")
        require(checked.returncode == 0, f"Gói giải nén không chạy được: {checked.stdout}\n{checked.stderr}")
    require(all(file_hash(path) == lock["artifacts"][name] for name, path in paths.items()), "Artifact đổi sau đóng gói.")
    receipt = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "status": "ready_for_nv3_handoff",
               "scope": "Four classical combinations, majority baseline, supplemental SVM; no PhoBERT training",
               "run_name": run_name, "models": names, "primary_model": primary_info,
               "dataset_version": manifest["dataset_version"], "split_manifest_sha256": split_hash,
               "sample_counts": {name: len(frame) for name, frame in frames.items()},
               "label_counts": manifest["label_counts"], "tests_log": str(test_log),
               "tests_count": int(re.search(r"Ran (\d+) tests", tests["stderr"]).group(1)),
               "test_metrics_recomputed": True, "validation_metrics_exported": True,
               "data_and_artifacts_unchanged": True, "anonymized_error_examples_verified": 25,
               "archive": archive.as_posix(), "archive_sha256": file_hash(archive),
               "archive_bytes": archive.stat().st_size, "portable_verify_stdout": checked.stdout,
               "portable_verify_stderr": checked.stderr, "portable_verified_without_real_data": True,
               "limitations": ["AI initial labels", "Previously used test; supplemental comparison", "No independent human test"]}
    write_json(output / "completion.json", receipt)
    print({"status": receipt["status"], "primary_model": primary, "archive": str(archive), "models": len(names)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="part3_final_20261102")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--test-log", required=True)
    args = parser.parse_args()
    finalize(args.run_name, args.output_dir, args.archive, args.test_log)


if __name__ == "__main__":
    run_cli(main)
