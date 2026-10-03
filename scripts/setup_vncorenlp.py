"""Tải riêng tài nguyên word segmentation; chỉ chạy khi người dùng cần PhoBERT."""
import argparse
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen
import shutil

from ptit_sentiment.common import file_hash, run_cli, write_json

FILES = ("VnCoreNLP-1.2.jar", "models/wordsegmenter/vi-vocab",
         "models/wordsegmenter/wordsegmenter.rdr")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="tools/vncorenlp")
    parser.add_argument("--revision", default="master", help="Commit VnCoreNLP để tái lập")
    args = parser.parse_args()
    directory = Path(args.output_dir).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Thư mục segmenter đã có dữ liệu; dùng bản local hoặc chọn thư mục mới.")
    records = []
    for name in FILES:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/{args.revision}/{name}"
        try:
            with urlopen(url, timeout=120) as response, path.with_suffix(path.suffix + ".part").open("wb") as stream:
                shutil.copyfileobj(response, stream)
            path.with_suffix(path.suffix + ".part").replace(path)
        except (URLError, OSError) as error:
            raise ValueError(f"Không tải được {url}: {error}. Chọn thư mục mới rồi thử lại.") from error
        records.append({"file": name, "url": url, "sha256": file_hash(path)})
        print(f"Đã tải: {path}")
    write_json(directory / "resources.json", {"revision": args.revision, "files": records})


if __name__ == "__main__":
    run_cli(main)
