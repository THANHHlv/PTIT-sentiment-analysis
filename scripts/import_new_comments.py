"""Nhập dữ liệu bình luận mới vào hệ thống thu thập.

Sử dụng khi người dùng xuất thêm bình luận từ Facebook (JSON/CSV/JSONL)
và muốn thêm vào bộ dữ liệu hiện có.

Cách dùng:
    $python = ".\.venv\Scripts\python.exe"
    # Bước 1: Đặt file JSON vào data/raw/browser_capture/ hoặc data/raw/incoming/
    # Bước 2: Chạy script này
    & $python scripts/import_new_comments.py --file data/raw/incoming/new_batch.json --source-id ptit_2k5
    # Bước 3: Kiểm tra report
"""
import argparse
import json
import sys
from pathlib import Path

# Thêm src vào path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ptit_sentiment.common import file_hash, load_config, run_cli, write_json


def validate_input_file(path):
    """Kiểm tra file đầu vào hợp lệ."""
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    if path.suffix.lower() not in (".json", ".jsonl", ".csv"):
        raise ValueError(f"Chỉ nhận JSON, JSONL hoặc CSV. File: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"File rỗng: {path}")
    return True


def preview_file(path, max_rows=5):
    """Xem trước nội dung file để xác nhận định dạng."""
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        rows = data if isinstance(data, list) else data.get("data", [])
        print(f"\nFile JSON: {len(rows)} bình luận")
        for i, row in enumerate(rows[:max_rows]):
            text_preview = str(row.get("text", ""))[:60]
            print(f"  [{i+1}] id={row.get('id','?')} post={row.get('post_id','?')} text=\"{text_preview}...\"")
        if len(rows) > max_rows:
            print(f"  ... và {len(rows) - max_rows} bình luận khác")
    elif path.suffix.lower() == ".csv":
        import csv
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        print(f"\nFile CSV: {len(rows)} bình luận")
        print(f"Cột: {list(rows[0].keys()) if rows else '(rỗng)'}")
        for i, row in enumerate(rows[:max_rows]):
            text_preview = str(row.get("text", ""))[:60]
            print(f"  [{i+1}] id={row.get('id','?')} post={row.get('post_id','?')} text=\"{text_preview}...\"")
    return True


def add_to_config(config_path, source_id, file_path):
    """Thêm file mới vào exports trong collection.yaml."""
    import yaml
    config = load_config(config_path)
    relative_path = str(file_path).replace("\\", "/")

    for source in config["sources"]:
        if source["source_id"] == source_id:
            exports = source.get("exports", [])
            if relative_path not in exports:
                exports.append(relative_path)
                source["exports"] = exports
                Path(config_path).write_text(
                    yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
                    encoding="utf-8"
                )
                print(f"Đã thêm {relative_path} vào exports của {source_id}")
            else:
                print(f"File {relative_path} đã có trong exports")
            return True

    raise ValueError(f"Không tìm thấy source_id '{source_id}' trong {config_path}. "
                     f"Source IDs có: {[s['source_id'] for s in config['sources']]}")


def import_and_collect(config_path, output_dir):
    """Chạy lại collect để nhập file mới."""
    from ptit_sentiment.data.collect import collect
    report = collect(config_path, output_dir)
    print(f"\nKết quả nhập:")
    print(f"  Tổng bình luận hợp lệ: {report['valid_unique_ids']}")
    print(f"  Số bài viết: {report['posts']}")
    print(f"  Đạt mục tiêu: {report['target_reached']}")
    if report.get("duplicate_content_groups"):
        print(f"  Nhóm nội dung trùng: {len(report['duplicate_content_groups'])}")
    if report.get("rejected_rows"):
        print(f"  Dòng bị loại: {len(report['rejected_rows'])}")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Nhập bình luận mới vào bộ thu thập PTIT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Nhập file JSON mới từ nhóm 2k5ptit
  python scripts/import_new_comments.py --file data/raw/incoming/batch003.json --source-id ptit_2k5

  # Nhập file CSV
  python scripts/import_new_comments.py --file data/raw/incoming/comments_new.csv --source-id ptit_group_584397217391365

  # Xem trước mà không nhập
  python scripts/import_new_comments.py --file data/raw/incoming/batch003.json --source-id ptit_2k5 --preview-only
"""
    )
    parser.add_argument("--file", required=True, help="Đường dẫn file JSON/CSV/JSONL cần nhập")
    parser.add_argument("--source-id", required=True,
                        help="ID nguồn trong collection.yaml (vd: ptit_2k5)")
    parser.add_argument("--config", default="configs/collection.yaml",
                        help="Đường dẫn file cấu hình thu thập")
    parser.add_argument("--output-dir", default="data/raw/ptit_sources_v1",
                        help="Thư mục lưu dữ liệu thu thập")
    parser.add_argument("--preview-only", action="store_true",
                        help="Chỉ xem trước, không nhập")
    args = parser.parse_args()

    file_path = Path(args.file)
    validate_input_file(file_path)
    preview_file(file_path)

    if args.preview_only:
        print("\n[Preview only] Không nhập dữ liệu.")
        return

    print(f"\nSHA256: {file_hash(file_path)}")
    add_to_config(args.config, args.source_id, file_path)
    report = import_and_collect(args.config, args.output_dir)

    if not report["target_reached"]:
        remaining = report.get("target_comments", 2000) - report["valid_unique_ids"]
        print(f"\n⚠ Còn thiếu ~{remaining} bình luận để đạt mục tiêu {report.get('target_comments', 2000)}.")
        print("  Thu thêm bình luận từ các bài viết khác trong nhóm PTIT.")


if __name__ == "__main__":
    run_cli(main)
