"""Nhập các đợt file xuất được phép dùng, checkpoint SQLite và nguồn truy vết."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from ptit_sentiment.common import file_hash, load_config, run_cli, write_json
from ptit_sentiment.data.validate import text_key

COLUMNS = ("id", "post_id", "text", "source_id", "source_url", "collected_at")


def export_rows(path, required_columns=()):
    """Đọc CSV/JSON/JSONL; không dò schema Facebook hoặc đọc thông tin tác giả."""
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            missing = set(required_columns) - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"CSV {path} thiếu cột: {sorted(missing)}.")
            yield from reader
    elif path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8-sig") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
    elif path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        rows = value.get("data") if isinstance(value, dict) else value
        if not isinstance(rows, list):
            raise ValueError("JSON phải là array hoặc object có data=array.")
        yield from rows
    else:
        raise ValueError("Chỉ nhận CSV, JSON hoặc JSONL; xem docs/real_data_workflow.md.")


def connect_store(directory):
    """SQLite transaction chứa checkpoint và dữ liệu; không ghi đè lần thu cũ."""
    directory.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(directory / "collection.sqlite3")
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS comments (
            id TEXT PRIMARY KEY, post_id TEXT NOT NULL, text TEXT NOT NULL,
            source_id TEXT, source_url TEXT, collected_at TEXT);
        CREATE TABLE IF NOT EXISTS checkpoints (
            file_key TEXT PRIMARY KEY, sha256 TEXT NOT NULL, row_offset INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS rejected (
            file_key TEXT, row_number INTEGER, reason TEXT, comment_id TEXT,
            PRIMARY KEY(file_key,row_number));
        CREATE TABLE IF NOT EXISTS duplicates (
            file_key TEXT, row_number INTEGER, comment_id TEXT,
            PRIMARY KEY(file_key,row_number));
    """)
    return db


def export_snapshot(db, directory):
    """CSV gốc tối thiểu từ transaction đã commit, thay tệp snapshot nguyên tử."""
    path = directory / "comments.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(COLUMNS)
        writer.writerows(db.execute("SELECT id,post_id,text,source_id,source_url,collected_at FROM comments ORDER BY id"))
    try:
        temporary.replace(path)
    except OSError:
        import shutil
        shutil.copyfile(temporary, path)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def import_source(db, source, directory, target, batch_size):
    """Tiếp tục theo vị trí dòng; hash thay đổi bị từ chối, cùng id khác nội dung bị cách ly."""
    if source.get("mode", "export") != "export":
        raise ValueError("Chưa có phương thức Facebook trực tiếp được xác nhận. Cần file xuất hoặc nguồn/API được phép; không vượt login/captcha.")
    if not source.get("source_id") or not source.get("source_url") or not source.get("permission_note"):
        raise ValueError("Nguồn cần source_id, source_url và permission_note.")
    if not source.get("exports"):
        raise ValueError("Nguồn chưa có exports (đường dẫn các đợt file xuất).")
    mapping = source.get("columns", {"id": "id", "post_id": "post_id", "text": "text"})
    for field in ("id", "post_id", "text"):
        if field not in mapping:
            raise ValueError(f"columns thiếu mapping {field}.")
    imported = 0
    seen_text_keys = set(text_key(row[0]) for row in db.execute("SELECT text FROM comments"))
    for filename in source["exports"]:
        path = Path(filename)
        digest = file_hash(path)
        key = source["source_id"] + ":" + str(path.resolve())
        checkpoint = db.execute("SELECT sha256,row_offset FROM checkpoints WHERE file_key=?", (key,)).fetchone()
        if checkpoint and checkpoint[0] != digest:
            raise ValueError(f"File xuất {path} đã đổi sau checkpoint; lưu bản mới bằng tên mới.")
        offset = checkpoint[1] if checkpoint else 0
        processed = offset
        for number, row in enumerate(export_rows(path, mapping.values()), start=1):
            if number <= offset:
                continue
            if db.execute("SELECT COUNT(*) FROM comments").fetchone()[0] >= target:
                break
            processed = number
            try:
                if not isinstance(row, dict):
                    raise ValueError("Dòng phải là object.")
                values = {field: row.get(mapping[field], "") for field in ("id", "post_id", "text")}
                if any(not isinstance(value, str) for value in values.values()):
                    raise ValueError("id/post_id/text cần chuỗi; JSON không dùng số cho mã.")
                if any(not value.strip() for value in values.values()):
                    raise ValueError("Thiếu id/post_id/text hoặc nội dung rỗng.")
                if any(values[field] != values[field].strip() for field in ("id", "post_id")):
                    raise ValueError("Mã có khoảng trắng đầu/cuối.")
                existing = db.execute("SELECT post_id,text FROM comments WHERE id=?", (values["id"],)).fetchone()
                tk = text_key(values["text"])
                if existing:
                    if existing != (values["post_id"], values["text"]):
                        raise ValueError("Cùng id nhưng post_id/text khác; cần kiểm tra nguồn.")
                    db.execute("INSERT OR IGNORE INTO duplicates VALUES(?,?,?)", (key, number, values["id"]))
                elif tk in seen_text_keys:
                    db.execute("INSERT OR IGNORE INTO duplicates VALUES(?,?,?)", (key, number, values["id"]))
                else:
                    seen_text_keys.add(tk)
                    now = datetime.now(timezone.utc).isoformat()
                    db.execute("INSERT INTO comments VALUES(?,?,?,?,?,?)",
                               (values["id"], values["post_id"], values["text"],
                                source["source_id"], source["source_url"], now))
                    imported += 1
            except ValueError as error:
                db.execute("INSERT OR REPLACE INTO rejected VALUES(?,?,?,?)",
                           (key, number, str(error), str(row.get(mapping["id"], "")) if isinstance(row, dict) else ""))
            if number % batch_size == 0:
                db.execute("INSERT OR REPLACE INTO checkpoints VALUES(?,?,?)", (key, digest, processed))
                db.commit()
                export_snapshot(db, directory)
        db.execute("INSERT OR REPLACE INTO checkpoints VALUES(?,?,?)", (key, digest, processed))
        db.commit()
        export_snapshot(db, directory)
    return imported


def collect(config_path, output_dir):
    """Lưu sau từng batch, chống trùng id, ghi nguồn lỗi và giữ số thật đã nhập."""
    config = load_config(config_path)
    if not config.get("sources"):
        raise ValueError("Chưa có nguồn PTIT trong cấu hình. Điền sources/exports; không có dữ liệu thật để thu.")
    target, batch = config.get("target_comments", 2000), config.get("batch_size", 250)
    if not isinstance(target, int) or target < 1 or not isinstance(batch, int) or batch < 1:
        raise ValueError("target_comments và batch_size phải là số nguyên dương.")
    source_ids = [source.get("source_id") for source in config["sources"]]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("source_id trùng trong cấu hình.")
    directory = Path(output_dir)
    db = connect_store(directory)
    sources = []
    try:
        for source in config["sources"]:
            info = {"source_id": source.get("source_id"), "source_url": source.get("source_url"),
                    "permission_note": source.get("permission_note"), "export_files": []}
            try:
                info["imported_this_run"] = import_source(db, source, directory, target, batch)
                info["status"] = "processed"
            except (ValueError, OSError, csv.Error, json.JSONDecodeError) as error:
                db.commit()  # Giữ phần đã nhập trước lỗi định dạng; không bịa đủ mục tiêu.
                info["status"] = "error"
                info["error"] = str(error)
            for filename in source.get("exports", []):
                path = Path(filename)
                if path.is_file():
                    info["export_files"].append({"path": str(path), "sha256": file_hash(path)})
            info["stored_comments"] = db.execute("SELECT COUNT(*) FROM comments WHERE source_id=?",
                                                  (source.get("source_id"),)).fetchone()[0]
            sources.append(info)
        export_snapshot(db, directory)
        rows = db.execute("SELECT id,post_id,text FROM comments").fetchall()
        keys = {}
        for identifier, _, text in rows:
            keys.setdefault(text_key(text), []).append(identifier)
        duplicates = [ids for ids in keys.values() if len(ids) > 1]
        report = {
            "provenance": "real_source_claimed_pending_review", "target_comments": target,
            "valid_unique_ids": len(rows), "posts": len({row[1] for row in rows}),
            "target_reached": len(rows) >= target,
            "sources": sources, "raw_csv_sha256": file_hash(directory / "comments.csv"),
            "duplicate_content_groups": duplicates,
            "rejected_rows": [dict(zip(("file_key", "row_number", "reason", "id"), row))
                              for row in db.execute("SELECT * FROM rejected")],
            "duplicate_id_rows": db.execute("SELECT COUNT(*) FROM duplicates").fetchone()[0],
            "label_confirmation": False,
        }
        write_json(directory / "collection_report.json", report)
        if any(info["status"] == "error" for info in sources):
            raise ValueError(f"Một số nguồn không nhập được; xem {directory / 'collection_report.json'}. Phần đã nhập được giữ để tiếp tục.")
        return report
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/collection.yaml")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = collect(args.config, args.output_dir)
    print({key: report[key] for key in ("valid_unique_ids", "posts", "target_reached")})


if __name__ == "__main__":
    run_cli(main)
