"""Kiểm tra tài nguyên local PhoBERT, không tải dữ liệu hay checkpoint."""
import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess

from ptit_sentiment.common import environment_versions, load_config, run_cli, write_json


def inspect_resources(config):
    """Trả các điều kiện đã phát hiện; không khẳng định runtime đã train được."""
    packages = {name: importlib.util.find_spec(name) is not None
                for name in ("torch", "transformers", "accelerate", "py_vncorenlp")}
    result = {"packages": packages, "versions": environment_versions()}
    result["java_available"] = shutil.which("java") is not None
    if result["java_available"]:
        proc = subprocess.run(["java", "-version"], capture_output=True, text=True)
        result["java_version"] = (proc.stderr or proc.stdout).strip()
    root = Path(config["segmenter_dir"])
    result["segmenter_files_present"] = all((root / name).is_file() for name in
        ("VnCoreNLP-1.2.jar", "models/wordsegmenter/vi-vocab", "models/wordsegmenter/wordsegmenter.rdr"))
    cache = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub"
    cached = cache / ("models--" + config["checkpoint"].replace("/", "--"))
    result["checkpoint_local_or_cached"] = Path(config["checkpoint"]).is_dir() or cached.is_dir()
    result["cuda_available"] = False
    if packages["torch"]:
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        result["torch_cuda_version"] = torch.version.cuda
        if result["cuda_available"]:
            result["gpu"] = torch.cuda.get_device_name(0)
            result["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    result["ready_for_offline_attempt"] = (
        all(packages.values()) and result["java_available"] and
        result["segmenter_files_present"] and result["checkpoint_local_or_cached"])
    result["runtime_training_verified"] = False
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phobert.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = inspect_resources(load_config(args.config))
    write_json(args.output, result)
    print(result)


if __name__ == "__main__":
    run_cli(main)
