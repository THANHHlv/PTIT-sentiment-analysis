"""Kiểm thử CLI PhoBERT thật: train ngắn, reload, evaluate và predict."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

import pandas as pd

from ptit_sentiment.common import read_json, run_cli, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/phobert_smoke.yaml")
    parser.add_argument("--skip-train", action="store_true", help="Kiểm tra artifact có validation_reference.json")
    args = parser.parse_args()
    output, artifact = Path(args.output_dir), Path(args.artifact_dir)
    if output.exists():
        raise ValueError("Thư mục output đã tồn tại; chọn lần chạy mới.")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "USE_TF": "0",
           "HF_HUB_DISABLE_TELEMETRY": "1"}
    env.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
    if not env.get("JAVA_HOME"):
        java = subprocess.run(["java", "-XshowSettings:properties", "-version"],
                              capture_output=True, text=True)
        match = re.search(r"java.home = (.+)", java.stderr)
        if match:
            env["JAVA_HOME"] = match.group(1).strip()
    output.mkdir(parents=True)
    commands = []

    def run(module, *options):
        command = [sys.executable, "-m", module, *map(str, options)]
        process = subprocess.run(command, env=env, capture_output=True, text=True, encoding="utf-8")
        commands.append({"command": command, "exit_code": process.returncode,
                         "stdout": process.stdout, "stderr": process.stderr})
        write_json(output / "commands.json", commands)
        if process.returncode:
            raise ValueError(f"Lệnh {module} thất bại: {process.stderr}")
        print(f"Đã chạy: {module}")

    if not args.skip_train:
        run("ptit_sentiment.training.train_phobert", "--splits-dir", args.splits_dir,
            "--config", args.config, "--artifact-dir", artifact)
    for split in ("validation", "test"):
        run("ptit_sentiment.evaluation.evaluate", "--model", artifact,
            "--splits-dir", args.splits_dir, "--split", split, "--output-dir", output / split)
    reference = read_json(artifact / "validation_reference.json")
    predicted = pd.read_csv(output / "validation/predictions.csv", dtype=str)
    if predicted["id"].tolist() != reference["ids"] or predicted["predicted_label"].tolist() != reference["predicted_labels"]:
        raise ValueError("Dự đoán validation thay đổi sau lưu/tải PhoBERT.")
    run("ptit_sentiment.predict", "--model", artifact, "--text", "Mình rất thích buổi học.",
        "--output", output / "sentence.csv")
    run("ptit_sentiment.predict", "--model", artifact, "--input", "tests/fixtures/new_comments.csv",
        "--output", output / "new_comments.csv")
    metadata = read_json(artifact / "metadata.json")
    write_json(output / "verification.json", {
        "status": "passed", "checkpoint": read_json(artifact / "metadata.json")["config"]["checkpoint"],
        "validation_reload_matches": True, "evaluation_and_prediction_cli": True,
        "training_config": metadata["config"]["training"],
        "provenance": metadata["provenance"],
        "not_real_ptit_experiment": metadata["provenance"] == "synthetic"})
    print(output)


if __name__ == "__main__":
    run_cli(main)
