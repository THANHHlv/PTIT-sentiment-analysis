"""Lưu bằng chứng nhỏ cho báo cáo; không sao chép text bình luận."""
import argparse
from pathlib import Path
import shutil

from ptit_sentiment.common import file_hash, read_json, run_cli, write_json
from ptit_sentiment.data.split import load_split_set


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root, output = Path(args.run_dir), Path(args.output_dir)
    if output.exists():
        raise ValueError("Thư mục bằng chứng đã có; chọn tên mới.")
    _, manifest, manifest_hash = load_split_set(args.splits_dir)
    paths = sorted((root / "evaluation").glob("*/metrics.json"))
    if not paths:
        raise ValueError("Chưa có metrics.json.")
    evidence = {"run": str(root), "provenance": manifest["provenance"],
                "source_sha256": manifest["source_sha256"], "manifest_sha256": manifest_hash,
                "splits": {name: {key: info[key] for key in ("rows", "posts", "label_counts", "sha256")}
                           for name, info in manifest["splits"].items()}, "models": []}
    for path in paths:
        result = read_json(path)
        if result["split_manifest_sha256"] != manifest_hash or result["split"] != "test":
            raise ValueError("Kết quả không thuộc test của manifest đã chọn.")
        evidence["models"].append({
            "source": str(path), "sha256": file_hash(path),
            **{key: result[key] for key in ("model_name", "accuracy", "macro_precision",
                                           "macro_recall", "macro_f1", "n_samples",
                                           "per_label", "confusion_matrix")}})
    output.mkdir(parents=True)
    write_json(output / "evidence.json", evidence)
    for name in ("comparison.csv", "comparison.png"):
        shutil.copyfile(root / name, output / name)
    shutil.copyfile(root / "training/training_summary.json", output / "training_summary.json")
    shutil.copyfile(root / "verification.json", output / "verification.json")
    print(f"Đã lưu bằng chứng {manifest['provenance']}: {output}")


if __name__ == "__main__":
    run_cli(main)
