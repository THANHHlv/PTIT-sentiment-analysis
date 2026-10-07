"""Chia/ghép nhãn của bốn người, phát hiện bất đồng và chờ xác nhận."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from ptit_sentiment.common import file_hash, read_json, run_cli, write_json
from ptit_sentiment.data.validate import read_csv, validate_frame
from ptit_sentiment.labels import LABELS

TASK_COLUMNS = ["id", "post_id", "text", "role", "suggested_label", "label", "confirmed", "notes"]


def prepare(input_path, output_dir, cross_fraction=0.2, seed=42, provenance="real", source_report=None):
    """Mỗi id có một người chính; một phần có thêm người độc lập, không mất dòng."""
    if not 0 < cross_fraction <= 1:
        raise ValueError("cross_fraction phải trong (0,1].")
    frame = read_csv(input_path, labeled=False)
    if "post_id" not in frame or frame["post_id"].str.strip().eq("").any():
        raise ValueError("Cần post_id thật cho mỗi bình luận, không tự tạo bài viết giả.")
    if frame["post_id"].str.strip().ne(frame["post_id"]).any():
        raise ValueError("post_id không có khoảng trắng đầu/cuối.")
    if provenance == "real":
        if not source_report:
            raise ValueError("Dữ liệu thật cần --source-report để truy vết nguồn.")
        report = read_json(source_report)
        if report["raw_csv_sha256"] != file_hash(input_path):
            raise ValueError("Nguồn CSV khác hash trong collection_report.")
        if not report["sources"] or any(source["status"] != "processed" for source in report["sources"]):
            raise ValueError("Nguồn cần được kiểm tra/nhập thành công trước gán nhãn.")
    directory = Path(output_dir)
    if directory.exists():
        raise ValueError("Thư mục phân công đã tồn tại; chọn phiên bản mới.")
    directory.mkdir(parents=True)
    base = frame[["id", "post_id", "text"]].copy()
    base.to_csv(directory / "base.csv", index=False, encoding="utf-8")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(base))
    assignments = {identifier: [] for identifier in base["id"]}
    tasks = {str(person): [] for person in range(1, 5)}
    cross_count = max(1, int(np.ceil(len(base) * cross_fraction)))
    cross_indices = set(rng.choice(len(base), size=cross_count, replace=False).tolist())
    for order, index in enumerate(indices):
        record = base.iloc[index].to_dict()
        primary = order % 4 + 1
        reviewers = [(primary, "primary")]
        if int(index) in cross_indices:
            secondary = (primary - 1 + (order % 3 + 1)) % 4 + 1
            reviewers.append((secondary, "cross_check"))
        for person, role in reviewers:
            assignments[record["id"]].append({"person": str(person), "role": role})
            tasks[str(person)].append({**record, "role": role, "suggested_label": "",
                                       "label": "", "confirmed": "", "notes": ""})
    for person, rows in tasks.items():
        pd.DataFrame(rows, columns=TASK_COLUMNS).to_csv(directory / f"annotator_{person}.csv", index=False, encoding="utf-8")
    manifest = {
        "input_sha256": file_hash(input_path), "base_sha256": file_hash(directory / "base.csv"),
        "source_report_sha256": file_hash(source_report) if source_report else None,
        "provenance": provenance, "seed": seed, "cross_fraction": cross_fraction,
        "samples": len(base), "cross_samples": cross_count,
        "assignments": assignments, "label_to_id_source": "ptit_sentiment.labels",
    }
    if source_report:
        write_json(directory / "source_report.json", read_json(source_report))
    write_json(directory / "assignment_manifest.json", manifest)
    return manifest


def read_tasks(directory, manifest, base):
    """Chỉ label/confirmed/notes/suggested_label được sửa; không nhận dòng ngoài phân công."""
    by_id = base.set_index("id")
    records, issues = {}, []
    for person in map(str, range(1, 5)):
        path = directory / f"annotator_{person}.csv"
        frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
        if set(TASK_COLUMNS) - set(frame.columns):
            raise ValueError(f"{path} thiếu cột bảng gán nhãn.")
        if frame["id"].duplicated().any():
            raise ValueError(f"{path} có id trùng.")
        expected = {identifier: next(a["role"] for a in assigned if a["person"] == person)
                    for identifier, assigned in manifest["assignments"].items()
                    if any(a["person"] == person for a in assigned)}
        if set(frame["id"]) - set(expected):
            raise ValueError(f"{path} có id ngoài phần phân công.")
        for identifier in set(expected) - set(frame["id"]):
            issues.append({"id": identifier, "reason": "missing_row", "person": person})
        for row in frame.to_dict(orient="records"):
            identifier = row["id"]
            if any(row[field] != by_id.loc[identifier, field] for field in ("post_id", "text")) or row["role"] != expected[identifier]:
                raise ValueError(f"{path}: đã sửa post_id/text/role của id {identifier}.")
            valid = row["label"] in LABELS and row["confirmed"].strip().lower() == "yes"
            records.setdefault(identifier, []).append({**row, "person": person, "valid": valid})
    return records, issues


def merge(assignment_dir, output_dir, resolutions=None, min_cross_agreement=0.8):
    """Xuất labeled.csv chỉ khi tất cả nhãn được xác nhận và bất đồng được phân xử."""
    if not 0 <= min_cross_agreement <= 1:
        raise ValueError("min_cross_agreement phải thuộc [0,1].")
    directory, output = Path(assignment_dir), Path(output_dir)
    if output.exists():
        raise ValueError("Thư mục ghép nhãn đã tồn tại; chọn phiên bản mới.")
    manifest = read_json(directory / "assignment_manifest.json")
    if file_hash(directory / "base.csv") != manifest["base_sha256"]:
        raise ValueError("base.csv đã bị sửa.")
    base = read_csv(directory / "base.csv", labeled=False)
    records, issues = read_tasks(directory, manifest, base)
    decisions = {}
    if resolutions:
        table = pd.read_csv(resolutions, dtype=str, encoding="utf-8-sig", keep_default_na=False)
        required = {"id", "final_label", "confirmed", "reviewer", "reason"}
        if required - set(table):
            raise ValueError(f"File phân xử cần {sorted(required)}.")
        if table["id"].duplicated().any() or set(table["id"]) - set(base["id"]):
            raise ValueError("File phân xử có id trùng hoặc ngoài dữ liệu.")
        decisions = {row["id"]: row for row in table.to_dict(orient="records")}
    labels, cross_pairs, unresolved = {}, [], []
    for identifier, assigned in manifest["assignments"].items():
        rows = records.get(identifier, [])
        valid = len(rows) == len(assigned) and all(row["valid"] for row in rows)
        agreed = valid and len({row["label"] for row in rows}) == 1
        if valid and len(rows) == 2:
            cross_pairs.append((rows[0]["label"], rows[1]["label"]))
        if agreed:
            labels[identifier] = rows[0]["label"]
            continue
        reason = "disagreement" if valid else "missing_invalid_or_unconfirmed_label"
        resolution = decisions.get(identifier)
        resolved = resolution and resolution["final_label"] in LABELS and resolution["confirmed"].strip().lower() == "yes"
        resolved = resolved and resolution["reviewer"].strip() and resolution["reason"].strip()
        if resolved:
            # Reviewer chỉ được phân xử khi các nhãn ban đầu đã được xác nhận.
            if valid:
                labels[identifier] = resolution["final_label"]
                continue
        unresolved.append({"id": identifier, "post_id": base.set_index("id").loc[identifier, "post_id"],
                           "issue_reason": reason, "labels": "|".join(row["label"] for row in rows),
                           "final_label": "", "confirmed": "", "reviewer": "", "reason": ""})
    completed_cross = len(cross_pairs)
    agreement = sum(a == b for a, b in cross_pairs) / completed_cross if completed_cross else None
    kappa = None
    if cross_pairs:
        value = float(cohen_kappa_score([a for a, _ in cross_pairs], [b for _, b in cross_pairs], labels=list(LABELS)))
        if np.isfinite(value):
            kappa = value
    cross_complete = completed_cross == manifest["cross_samples"]
    quality_review_required = not cross_complete or agreement is None or agreement < min_cross_agreement
    # Ngưỡng thấp yêu cầu xem lại toàn quy tắc, không chỉ sửa vài bất đồng.
    ready = not unresolved and not issues and not quality_review_required
    output.mkdir(parents=True)
    pd.DataFrame(unresolved, columns=["id", "post_id", "issue_reason", "labels", "final_label", "confirmed",
                                    "reviewer", "reason"]).to_csv(output / "needs_review.csv", index=False, encoding="utf-8")
    report = {
        "status": "confirmed" if ready else "needs_human_review",
        "provenance": manifest["provenance"], "samples": len(base),
        "confirmed_samples": len(labels), "unresolved_samples": len(unresolved), "issues": issues,
        "cross_samples_planned": manifest["cross_samples"], "cross_samples_completed": completed_cross,
        "cross_agreement": agreement, "pooled_pair_cohen_kappa": kappa,
        "kappa_note": "Pairs from different reviewer combinations; descriptive pooled diagnostic, not a multi-rater coefficient.",
        "min_cross_agreement": min_cross_agreement, "quality_review_required": quality_review_required,
        "assignment_manifest_sha256": file_hash(directory / "assignment_manifest.json"),
        "base_sha256": manifest["base_sha256"], "source_report_sha256": manifest["source_report_sha256"],
        "annotator_file_hashes": {str(person): file_hash(directory / f"annotator_{person}.csv") for person in range(1, 5)},
        "resolutions_sha256": file_hash(resolutions) if resolutions else None,
        "label_counts": {label: sum(value == label for value in labels.values()) for label in LABELS},
    }
    if ready:
        labeled = base.copy()
        labeled["label"] = labeled["id"].map(labels)
        validate_frame(labeled)
        labeled.to_csv(output / "labeled.csv", index=False, encoding="utf-8")
        report["labeled_sha256"] = file_hash(output / "labeled.csv")
    write_json(output / "labeling_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--input", required=True)
    prep.add_argument("--output-dir", required=True)
    prep.add_argument("--cross-fraction", type=float, default=0.2)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--provenance", choices=("real", "synthetic"), default="real")
    prep.add_argument("--source-report")
    merged = commands.add_parser("merge")
    merged.add_argument("--assignment-dir", required=True)
    merged.add_argument("--output-dir", required=True)
    merged.add_argument("--resolutions")
    merged.add_argument("--min-cross-agreement", type=float, default=0.8)
    args = parser.parse_args()
    if args.command == "prepare":
        report = prepare(args.input, args.output_dir, args.cross_fraction, args.seed, args.provenance, args.source_report)
        print(f"Đã phân công {report['samples']} câu; {report['cross_samples']} câu kiểm tra chéo.")
    else:
        report = merge(args.assignment_dir, args.output_dir, args.resolutions, args.min_cross_agreement)
        print(report)
        if report["status"] != "confirmed":
            raise ValueError(f"Chưa đủ nhãn chuẩn; xem {Path(args.output_dir) / 'needs_review.csv'} và labeling_report.json.")


if __name__ == "__main__":
    run_cli(main)
