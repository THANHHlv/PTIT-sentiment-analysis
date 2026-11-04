"""Run the real-data NV2 audit, CPU classical search, locked test evaluation and checks."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from ptit_sentiment.common import file_hash, read_json, run_cli, write_json
from ptit_sentiment.data.split import load_split_set
from ptit_sentiment.evaluation.metrics import compute_metrics
from ptit_sentiment.models.classical import MODEL_NAMES, predict_classical
from ptit_sentiment.predict import load_artifact, predict_with_artifact


def run(run_name, config, splits_dir, extended_config=None, resume=False):
    """Keep each run isolated; save CLI logs, selected models and test evidence."""
    if not run_name or Path(run_name).name != run_name or run_name in (".", ".."):
        raise ValueError("run-name must be a directory name, without path separators.")
    output, artifact = Path("outputs") / run_name, Path("artifacts") / run_name
    if not resume and (artifact.exists() or (output.exists() and any(output.iterdir()))):
        raise ValueError("Run already exists; choose a new --run-name.")
    if resume and not (output / "run_context.json").exists():
        raise ValueError("Cannot resume without the original run_context.json.")
    if resume and (output / "verification.json").exists():
        raise ValueError("Run already completed; its evidence must be preserved.")
    output.mkdir(parents=True, exist_ok=resume)
    commands = read_json(output / "commands.json") if resume else []
    started = (read_json(output / "run_context.json")["started_at_utc"] if resume
               else datetime.now(timezone.utc).isoformat())
    clock = perf_counter()
    def command(*arguments, expected_success=True):
        cmd = [sys.executable, *map(str, arguments)]
        if resume and any(item["command"] == cmd and
                          (item["returncode"] == 0) == expected_success for item in commands):
            print(f"Preserved completed step: {' '.join(map(str, arguments))}", flush=True)
            return
        step_clock = perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        commands.append({"command": cmd, "returncode": result.returncode,
                         "stdout": result.stdout, "stderr": result.stderr,
                         "seconds": perf_counter() - step_clock})
        write_json(output / "commands.json", commands)
        if (result.returncode == 0) != expected_success:
            raise ValueError(f"Unexpected CLI result: {cmd}: {result.stdout}\n{result.stderr}")
        print(f"Checked: {' '.join(map(str, arguments))}", flush=True)
    code_files = [*Path("src").rglob("*.py"), Path(config), Path("features.py"),
                  Path("scripts/review_part2.py"), Path("scripts/run_part3.py"),
                  Path("scripts/audit_selected_features.py")]
    if extended_config:
        code_files.append(Path(extended_config))
    context = {
        "started_at_utc": started, "code_sha256": {str(p): file_hash(p) for p in code_files},
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "git_status": subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout,
        "policy": "CPU; AI initial labels; no relabel/split/refit; validation selection before test; test previously used",
    }
    if resume:
        original = read_json(output / "run_context.json")
        for path, digest_before in original["code_sha256"].items():
            # Only the orchestration runner may change to add resume support.
            if Path(path) != Path("scripts/run_part3.py") and file_hash(path) != digest_before:
                raise ValueError(f"Cannot resume after code/config changes: {path}")
        context["resumed_at_utc"] = datetime.now(timezone.utc).isoformat()
        context["reason"] = "Continue interrupted orchestration; preserve trained selections and command logs"
        resume_log = output / ("resume_context_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + ".json")
        write_json(resume_log, context)
    else:
        write_json(output / "run_context.json", context)
    command("scripts/review_part2.py", "--splits-dir", splits_dir, "--config", config,
            "--output", output / "part2_audit.json")
    frames, manifest, digest = load_split_set(splits_dir)
    if manifest.get("provenance") != "real":
        raise ValueError("This runner requires real data; never substitute fixtures.")
    command("-m", "unittest", "discover", "-s", "tests", "-v")
    command("-m", "ptit_sentiment.training.train_classical", "--splits-dir", splits_dir,
            "--config", config, "--artifact-dir", artifact, "--output-dir", output / "training")
    lock = read_json(output / "training/selection_lock.json")
    assert not lock["test_predictions_computed"] and lock["split_manifest_sha256"] == digest
    names = (*MODEL_NAMES, "majority_baseline")
    artifacts = {name: artifact / (name + ".joblib") for name in names}
    locks = {output / "training/selection_lock.json": file_hash(output / "training/selection_lock.json")}
    hashes = dict(lock["artifacts"])
    for name in names:
        assert file_hash(artifacts[name]) == hashes[name]
    if extended_config:
        command("-m", "ptit_sentiment.training.train_classical", "--splits-dir", splits_dir,
                "--config", extended_config, "--artifact-dir", artifact / "extended",
                "--output-dir", output / "training_extended")
        extended_lock_path = output / "training_extended/selection_lock.json"
        extended_lock = read_json(extended_lock_path)
        assert not extended_lock["test_predictions_computed"]
        assert extended_lock["split_manifest_sha256"] == digest
        locks[extended_lock_path] = file_hash(extended_lock_path)
        for selected in extended_lock["selected_models"]:
            name = selected["model_name"]
            if name in artifacts:
                raise ValueError(f"Supplemental model needs a unique alias: {name}")
            artifacts[name] = artifact / "extended" / (name + ".joblib")
            hashes[name] = extended_lock["artifacts"][name]
            assert file_hash(artifacts[name]) == hashes[name]
    names = tuple(artifacts)
    # Audit every selected representation using only train and validation.
    for name in names:
        if name != "majority_baseline":
            command("scripts/audit_selected_features.py", "--model", artifacts[name],
                    "--splits-dir", splits_dir, "--output", output / "feature_audits" / (name + ".json"))
    write_json(output / "all_selection_lock.json", {
        "locked_at_utc": datetime.now(timezone.utc).isoformat(), "split_manifest_sha256": digest,
        "artifacts": hashes, "selection_locks": {str(p): h for p, h in locks.items()},
        "test_predictions_computed": False, "test_previously_used": True,
        "evaluation_scope": "Supplemental comparison on the existing AI-labeled test, not a fresh independent test",
    })
    all_lock_hash = file_hash(output / "all_selection_lock.json")
    # All core and supplemental selections are locked before the first test call.
    for name in names:
        command("-m", "ptit_sentiment.evaluation.evaluate", "--model", artifacts[name],
                "--splits-dir", splits_dir, "--output-dir", output / "evaluation" / name)
    command("-m", "ptit_sentiment.evaluation.compare", "--inputs",
            *(output / "evaluation" / name / "metrics.json" for name in names),
            "--output", output / "comparison.csv")
    command("-m", "ptit_sentiment.predict", "--model", artifact / "tfidf_svm.joblib",
            "--text", "Mình chưa hài lòng với lịch học.", "--output", output / "single_prediction.csv")
    command("-m", "ptit_sentiment.predict", "--model", artifact / "bow_nb.joblib",
            "--input", "tests/fixtures/new_comments.csv", "--output", output / "new_predictions.csv")
    command("-m", "ptit_sentiment.predict", "--model", artifact / "bow_nb.joblib",
            "--text", " ", expected_success=False)
    command("-m", "ptit_sentiment.predict", "--model", artifact / "missing.joblib",
            "--text", "Bình luận mới", expected_success=False)
    verification = {}
    test = frames["test"]
    for name in names:
        model, metadata = load_artifact(artifacts[name])
        actual = predict_with_artifact(model, metadata, test)
        saved = pd.read_csv(output / "evaluation" / name / "predictions.csv", dtype=str, keep_default_na=False)
        assert actual["id"].tolist() == saved["id"].tolist() == test["id"].tolist()
        assert saved["text"].tolist() == test["text"].tolist()
        np.testing.assert_array_equal(saved["true_label"], test["label"])
        np.testing.assert_array_equal(saved["predicted_label"], actual["predicted_label"])
        second = joblib.load(artifacts[name])
        predicted, _, _ = predict_classical(second, test["text"], name == "majority_baseline")
        np.testing.assert_array_equal(predicted, saved["predicted_label"])
        metrics = read_json(output / "evaluation" / name / "metrics.json")
        recomputed = compute_metrics(saved["true_label"].tolist(), saved["predicted_label"].tolist())
        assert all(metrics[k] == v for k, v in recomputed.items())
        errors = pd.read_csv(output / "evaluation" / name / "errors.csv")
        assert len(errors) == int((saved.true_label != saved.predicted_label).sum())
        assert metadata["artifact_sha256"] == hashes[name]
        verification[name] = {"row_order_labels_text_equal": True, "reload_equal_on_all_test": True,
                              "metrics_recomputed_equal": True, "errors": len(errors)}
    audit = read_json(output / "part2_audit.json")
    assert all(file_hash(path) == value for path, value in audit["files_unchanged"].items())
    assert all(value == file_hash(path) for path, value in locks.items())
    assert all_lock_hash == file_hash(output / "all_selection_lock.json")
    write_json(output / "verification.json", {
        "started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "seconds": perf_counter() - clock, "split_manifest_sha256": digest,
        "selection_lock_sha256": all_lock_hash, "data_unchanged": True,
        "all_models_locked_before_test": True, "models": verification,
        "cli_one_comment_csv_empty_missing_artifact": True,
        "tests_are_synthetic_only": True, "evaluation_is_real_ai_labels": True,
        "test_previously_used": True,
    })
    print(pd.read_csv(output / "comparison.csv").to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="part3_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--config", default="configs/part3.yaml")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--extended-config", help="Optional supplemental search; locked before all test evaluations")
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted run without repeating completed CLI steps")
    args = parser.parse_args()
    run(args.run_name, args.config, args.splits_dir, args.extended_config, args.resume)


if __name__ == "__main__":
    run_cli(main)
