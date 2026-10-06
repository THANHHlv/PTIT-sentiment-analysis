"""Xác minh đợt DOM Chrome, cập nhật exports và nhập có checkpoint."""
import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import yaml
from ptit_sentiment.common import file_hash, load_config, run_cli, write_json
from ptit_sentiment.data.collect import collect

SOURCE_PREFIXES = {"ptit_2k5": "ptit_2k5", "ptit_group_584397217391365": "ptit_info"}
ALLOWED_GROUPS = {"ptit_2k5": {"2k5ptit", "450310163844833"},
                  "ptit_group_584397217391365": {"584397217391365"}}


def sync(config_path, directory, output_dir, target=None):
    """Chỉ nhận nội dung đầy đủ và permalink DOM phù hợp nguồn; không tự tạo nhãn."""
    import json
    config = load_config(config_path)
    if target is not None and int(target) > 0:
        config["target_comments"] = int(target)
    capture_dir = Path(directory)
    audit = []
    for source in config["sources"]:
        source_id = source["source_id"]
        prefix = SOURCE_PREFIXES[source_id]
        paths = sorted(capture_dir.glob(f"{prefix}_snapshot*.json"))
        for path in paths:
            rows = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(rows, list) or not rows:
                raise ValueError(f"{path}: cần danh sách bình luận không rỗng.")
            for row in rows:
                if row.get("is_truncated") or not row.get("text", "").strip():
                    raise ValueError(f"{path}: nội dung rỗng hoặc còn rút gọn.")
                parsed = urlparse(row.get("source_url", ""))
                parts = parsed.path.strip("/").split("/")
                valid = (parsed.scheme == "https" and parsed.hostname == "www.facebook.com"
                         and len(parts) == 4 and parts[0] == "groups" and parts[2] == "posts"
                         and parts[1] in ALLOWED_GROUPS[source_id] and parts[3] == row["post_id"])
                if not valid:
                    raise ValueError(f"{path}: permalink không khớp nguồn/post_id.")
                if not isinstance(row["id"], str) or not row["id"].isdigit():
                    raise ValueError(f"{path}: mã bình luận DOM phải là chuỗi số.")
            audit.append({"source_id": source_id, "path": str(path), "rows": len(rows), "sha256": file_hash(path)})
        if paths:
            source["exports"] = [str(path).replace("\\", "/") for path in paths]
            source["permission_note"] = "Người dùng chỉ định nhóm và kết nối Chrome; đọc DOM hiển thị bằng phiên sẵn có, không lấy cookie/token hoặc thông tin tác giả."
    Path(config_path).write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    write_json(Path(output_dir) / "browser_capture_audit.json", {"method": "rendered DOM via connected Chrome",
               "batches": audit, "labels_confirmed": False,
               "limitations": ["Only loaded text comments; image/sticker-only comments excluded",
                               "Group membership does not verify student status of each author",
                               "Text is rendered DOM content, not an API export",
                               "Raw text may contain contact details; excluded from Git, mask before sharing"]})
    return collect(config_path, output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/collection.yaml")
    parser.add_argument("--capture-dir", default="data/raw/browser_capture")
    parser.add_argument("--output-dir", default="data/raw/ptit_sources_v1")
    args = parser.parse_args()
    report = sync(args.config, args.capture_dir, args.output_dir)
    print({key: report[key] for key in ("valid_unique_ids", "posts", "target_reached")})


if __name__ == "__main__":
    run_cli(main)