"""Verify an extracted NV3 handoff and predict with all saved models, without training."""
import argparse
from pathlib import Path

import pandas as pd

from ptit_sentiment.common import file_hash, read_json, run_cli
from ptit_sentiment.predict import load_artifact, predict_with_artifact


def bundle_path(root, relative):
    """Resolve a manifest member and reject references outside the handoff root."""
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or Path(relative).is_absolute():
        raise ValueError(f"Đường dẫn ngoài gói bàn giao: {relative}")
    return path


def verify_bundle(root):
    """Check every member hash, then compare saved-model predictions with the receipt."""
    root = Path(root).resolve()
    manifest = read_json(root / "bundle_manifest.json")
    for relative, digest in manifest["files_sha256"].items():
        if file_hash(bundle_path(root, relative)) != digest:
            raise ValueError(f"SHA-256 không khớp: {relative}")
    frame = pd.DataFrame(manifest["prediction_check"]["inputs"])
    for name, relative in manifest["models"].items():
        model, metadata = load_artifact(bundle_path(root, relative))
        actual = predict_with_artifact(model, metadata, frame)
        if actual["predicted_label"].tolist() != manifest["prediction_check"]["expected"][name]:
            raise ValueError(f"Dự đoán thay đổi sau đóng gói: {name}")
        if set(actual["model_name"]) != {name}:
            raise ValueError(f"Tên mô hình không khớp: {name}")
    return {"files_verified": len(manifest["files_sha256"]),
            "models_verified": len(manifest["models"]),
            "primary_model": manifest["primary_model"], "prediction_equal": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Thư mục đã giải nén gói NV3")
    args = parser.parse_args()
    print(verify_bundle(args.root))


if __name__ == "__main__":
    run_cli(main)
