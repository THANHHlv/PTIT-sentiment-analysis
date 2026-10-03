"""Đọc cấu hình, JSON, mã băm và thông báo lỗi CLI dùng chung."""

import hashlib
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml


def load_config(path):
    """Đọc YAML mapping; báo lỗi nếu tệp không hợp lệ."""
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Cấu hình {path} phải là mapping YAML.")
    return config


def write_json(path, value):
    """Ghi JSON UTF-8; tự tạo thư mục cha."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    """SHA256 byte của tệp để phát hiện thay đổi dữ liệu."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment_versions():
    """Ghi phiên bản thực tế, không import thư viện mô hình nặng."""
    result = {}
    for package in ("ptit-sentiment", "scikit-learn", "pandas", "numpy", "underthesea",
                    "transformers", "torch", "py-vncorenlp"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            pass
    return result


def run_cli(main):
    """Hiển thị lỗi dữ liệu/cấu hình/tệp gọn và trả exit code khác 0."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        main()
    except (ValueError, FileNotFoundError, ImportError, KeyError, OSError) as error:
        raise SystemExit(f"Lỗi: {error}") from error
