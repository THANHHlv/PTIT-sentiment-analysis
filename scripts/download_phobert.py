"""Tải checkpoint PhoBERT công khai vào cache trong workspace, không gửi dữ liệu."""
import argparse
import os
from pathlib import Path

from ptit_sentiment.common import load_config, run_cli, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phobert.yaml")
    parser.add_argument("--cache-dir", default=".cache/huggingface/hub")
    args = parser.parse_args()
    config = load_config(args.config)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download
    directory = snapshot_download(
        config["checkpoint"], revision=config["revision"], cache_dir=args.cache_dir,
        allow_patterns=["config.json", "vocab.txt", "bpe.codes", "tokenizer_config.json",
                        "special_tokens_map.json", "pytorch_model.bin"], max_workers=2)
    write_json(Path(args.cache_dir).parent / "download_manifest.json",
               {"checkpoint": config["checkpoint"], "revision": config["revision"],
                "snapshot": directory, "resolved_commit": Path(directory).name})
    print(directory)


if __name__ == "__main__":
    run_cli(main)
