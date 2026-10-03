"""Chia nhóm post_id; không bao giờ tự chuyển sang chia theo bình luận."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ptit_sentiment.common import file_hash, load_config, read_json, run_cli, write_json
from ptit_sentiment.data.validate import read_csv, statistics, text_key
from ptit_sentiment.labels import LABELS

SPLIT_NAMES = ("train", "validation", "test")


def check_disjoint(frames):
    """Kiểm tra nhãn đầy đủ và giao id, post_id, khóa text giữa các tập."""
    for name, frame in frames.items():
        if set(frame["label"]) != set(LABELS):
            raise ValueError(f"Tập {name} thiếu nhãn; cần đủ {LABELS}.")
    names = list(frames)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            for col in ("id", "post_id", "text"):
                a, b = frames[left][col], frames[right][col]
                if col == "text":
                    a, b = a.map(text_key), b.map(text_key)
                overlap = set(a) & set(b)
                if overlap:
                    raise ValueError(f"Rò rỉ {col} giữa {left}/{right}: {list(overlap)[:5]}.")


def grouped_split(frame, ratios=(0.7, 0.15, 0.15), seed=42, attempts=2000):
    """Tìm bộ chia nhóm gần tỷ lệ dòng/nhãn mục tiêu bằng seed cố định.

    Tỷ lệ xấp xỉ vì nhóm bài viết là đơn vị không thể chia nhỏ.
    Nếu không tìm được đủ ba nhãn, báo lỗi; không fallback gây rò rỉ.
    """
    ratios = np.asarray(ratios, dtype=float)
    if ratios.shape != (3,) or not np.isfinite(ratios).all() or (ratios <= 0).any() or not np.isclose(ratios.sum(), 1):
        raise ValueError("ratios phải gồm ba số dương có tổng bằng 1.")
    if attempts < 1:
        raise ValueError("attempts phải >= 1.")
    counts = pd.crosstab(frame["post_id"], frame["label"]).reindex(columns=LABELS, fill_value=0)
    if len(counts) < 3 or ((counts > 0).sum(axis=0) < 3).any():
        raise ValueError("Dữ liệu quá nhỏ: mỗi nhãn phải xuất hiện trong ít nhất 3 post_id khác nhau.")
    group_sizes = np.maximum(1, np.floor(ratios * len(counts)).astype(int))
    while group_sizes.sum() < len(counts):
        group_sizes[np.argmax(ratios * len(counts) - group_sizes)] += 1
    while group_sizes.sum() > len(counts):
        eligible = np.where(group_sizes > 1)[0]
        if not len(eligible):
            raise ValueError("Không đủ bài viết cho ba tập.")
        index = eligible[np.argmax(group_sizes[eligible] - ratios[eligible] * len(counts))]
        group_sizes[index] -= 1
    rng = np.random.default_rng(seed)
    matrix, groups = counts.to_numpy(), counts.index.to_numpy()
    totals = matrix.sum(axis=0)
    best, best_score = None, float("inf")
    boundaries = np.cumsum(group_sizes)[:-1]
    for _ in range(attempts):
        candidate = np.split(rng.permutation(len(groups)), boundaries)
        supports = np.array([matrix[indices].sum(axis=0) for indices in candidate])
        if (supports == 0).any():
            continue
        score = float(np.abs(supports / totals - ratios[:, None]).sum()
                      + np.abs(supports.sum(axis=1) / len(frame) - ratios).sum())
        if score < best_score:
            best, best_score = candidate, score
    if best is None:
        raise ValueError(f"Không tìm được bộ chia nhóm đủ ba nhãn sau {attempts} lần. "
                         "Bổ sung dữ liệu/nhóm hoặc điều chỉnh ratios/attempts; không chia ngẫu nhiên theo dòng.")
    frames = {
        name: frame[frame["post_id"].isin(groups[indices])].copy().reset_index(drop=True)
        for name, indices in zip(SPLIT_NAMES, best)
    }
    check_disjoint(frames)
    return frames


def save_splits(input_path, output_dir, ratios, seed, attempts, provenance):
    """Lưu CSV nguyên text, metadata, hash và danh sách thành viên nhóm/id."""
    frame = read_csv(input_path)
    frames = grouped_split(frame, ratios, seed, attempts)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Thư mục {output_dir} đã có dữ liệu; hãy chọn thư mục mới để giữ bộ chia cũ.")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1, "source": str(Path(input_path).resolve()),
        "source_sha256": file_hash(input_path), "provenance": provenance,
        "method": "seeded_group_search_post_id_v1", "seed": seed,
        "requested_ratios": list(ratios), "attempts": attempts, "splits": {},
    }
    for name, subset in frames.items():
        path = output_dir / f"{name}.csv"
        subset.to_csv(path, index=False, encoding="utf-8")
        manifest["splits"][name] = {
            **statistics(subset), "actual_ratio": len(subset) / len(frame),
            "sha256": file_hash(path), "ids": subset["id"].tolist(),
            "post_ids": sorted(set(subset["post_id"])),
        }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def load_split_set(directory):
    """Đọc bộ chia và xác minh hash, thành viên, nhãn và không rò rỉ."""
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    frames = {}
    for name in SPLIT_NAMES:
        path = directory / f"{name}.csv"
        info = manifest["splits"][name]
        if file_hash(path) != info["sha256"]:
            raise ValueError(f"{path} đã đổi so với manifest; tạo bộ chia mới, không sửa tập hiện có.")
        frame = read_csv(path)
        if frame["id"].tolist() != info["ids"] or sorted(set(frame["post_id"])) != info["post_ids"]:
            raise ValueError(f"Thành viên {name} khác manifest.")
        frames[name] = frame
    check_disjoint(frames)
    return frames, manifest, file_hash(directory / "manifest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/classical.yaml")
    parser.add_argument("--provenance", required=True, choices=("synthetic", "real"))
    args = parser.parse_args()
    config = load_config(args.config)["split"]
    manifest = save_splits(args.input, args.output_dir, config["ratios"],
                           config["seed"], config.get("attempts", 2000), args.provenance)
    print({name: info for name, info in manifest["splits"].items()
           if name in SPLIT_NAMES})


if __name__ == "__main__":
    run_cli(main)
