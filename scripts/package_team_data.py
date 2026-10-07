"""Đóng gói bộ chia đã kiểm tra để bàn giao nội bộ; không tải lên dịch vụ nào.

Đầu vào: data/labeled và data/splits của cùng phiên bản nhãn AI ban đầu.
Đầu ra: ZIP giữ đường dẫn data/... và bảng SHA-256, không gồm raw/cache/model.
Nội dung bình luận được giữ nguyên; gói này chưa được ẩn danh để công khai.
"""
import argparse
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from ptit_sentiment.common import file_hash, read_json, run_cli
from ptit_sentiment.data.ai_dataset import checked_origins
from ptit_sentiment.data.split import load_split_set
from ptit_sentiment.data.validate import read_csv


def package_data(root, output=None):
    """Kiểm tra hash/nguồn nhãn rồi tạo ZIP mới; từ chối ghi đè tệp đã có."""
    root = Path(root).resolve()
    splits, labeled = root / 'data/splits', root / 'data/labeled'
    frames, manifest, digest = load_split_set(splits)
    if manifest.get('labeling_policy') != 'ai_initial':
        raise ValueError('Công cụ này chỉ bàn giao bộ AI ban đầu đã ghi rõ nguồn nhãn.')
    source = read_csv(labeled / 'labeled_comments.csv')
    checked_origins(source)
    if file_hash(labeled / 'labeled_comments.csv') != manifest['source_sha256']:
        raise ValueError('labeled_comments.csv khác nguồn trong manifest.')
    joined = pd.concat(frames.values(), ignore_index=True)
    columns = list(source.columns)
    if set(joined.columns) != set(columns):
        raise ValueError('Cột nguồn và bộ chia không khớp.')
    if not source.sort_values('id').reset_index(drop=True).equals(
            joined[columns].sort_values('id').reset_index(drop=True)):
        raise ValueError('Nội dung nguồn và bộ chia không khớp.')
    compact = manifest.get('storage_layout') == 'compact_v1'
    if compact:
        audit = manifest['embedded_metadata']['labeling_report']
        if len(source) != audit['labeled_rows']:
            raise ValueError('Số dòng nhãn khác báo cáo nhúng.')
        paths = [labeled / 'labeled_comments.csv'] + [splits / n for n in
                 ('train.csv', 'validation.csv', 'test.csv', 'manifest.json')]
    else:
        audit = read_json(labeled / 'ai_labeling_report.json')
        if file_hash(labeled / 'ai_labeling_report.json') != manifest['labeling_report_sha256']:
            raise ValueError('Báo cáo gán nhãn khác phiên bản bộ chia.')
        unresolved = pd.read_csv(labeled / 'unresolved.csv', dtype=str, keep_default_na=False)
        excluded = pd.read_csv(labeled / 'excluded_comments.csv', dtype=str, keep_default_na=False)
        if (len(source), len(unresolved), len(excluded)) != (
                audit['labeled_rows'], audit['unresolved_rows'], audit['excluded_rows']):
            raise ValueError('Số dòng labeled/unresolved/excluded khác báo cáo.')
        if not (unresolved['label'].eq('').all()
                and unresolved['annotation_status'].eq('unresolved').all()):
            raise ValueError('unresolved chứa nhãn hoặc trạng thái không hợp lệ.')
        combined = pd.concat([source, unresolved, excluded], ignore_index=True)
        if len(combined) != audit['initial_rows'] or combined['id'].duplicated().any():
            raise ValueError('Đối soát tổng số dòng/ID ba nhóm thất bại.')
        reconciliation = pd.read_csv(labeled / 'id_reconciliation.csv', dtype=str, keep_default_na=False)
        expected = {row_id: name for name, table in (
            ('labeled', source), ('unresolved', unresolved), ('excluded', excluded))
                    for row_id in table['id']}
        if (reconciliation['id'].duplicated().any()
                or dict(zip(reconciliation['id'], reconciliation['destination'])) != expected):
            raise ValueError('id_reconciliation không khớp ba nhóm dữ liệu.')
        paths = [splits / name for name in (
            'train.csv', 'validation.csv', 'test.csv', 'manifest.json', 'metadata.json',
            'labeling_report.json', 'split_groups.csv', 'near_duplicate_constraints.csv')]
        paths += [labeled / name for name in (
            'labeled_comments.csv', 'ai_labeling_report.json', 'unresolved.csv',
            'excluded_comments.csv', 'id_reconciliation.csv')]
        if file_hash(splits / 'metadata.json') != digest:
            raise ValueError('metadata.json và manifest.json không khớp.')
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f'Thiếu tệp bàn giao: {path.name}')
    version = manifest['dataset_version']
    summary = {
        'dataset_version': version, 'manifest_sha256': digest,
        'labeling_policy': 'ai_initial', 'human_labels_confirmed': False,
        'initial_rows': audit['initial_rows'], 'labeled_rows': len(source),
        'unresolved_rows': audit['unresolved_rows'], 'excluded_rows': audit['excluded_rows'],
        'label_counts': manifest['label_counts'],
        'splits': {n: {k: info[k] for k in ('rows', 'posts', 'label_counts', 'sha256')}
                   for n, info in manifest['splits'].items()},
        'files_sha256': {p.relative_to(root).as_posix(): file_hash(p) for p in paths},
        'distribution': 'Internal team only; contains original comment text and source identifiers.',
        'limitation': manifest['limitation'],
        'storage_layout': manifest.get('storage_layout', 'full'),
        'source_audit_note': 'Original source audit archived separately.' if compact else 'Included reconciliation of all input IDs.',
    }
    destination = Path(output) if output else root / 'handoff' / f'{version}.zip'
    if not destination.is_absolute():
        destination = root / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Mode x bảo vệ gói đã chốt; không sửa manifest hoặc nội dung CSV.
    with ZipFile(destination, 'x', compression=ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr('TEAM_DATA.json', json.dumps(summary, ensure_ascii=False, indent=2))
    return destination, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--output', help='ZIP mới; mặc định handoff/<dataset_version>.zip')
    args = parser.parse_args()
    destination, summary = package_data(args.root, args.output)
    print(json.dumps({'zip': str(destination), 'sha256': file_hash(destination),
                      'dataset_version': summary['dataset_version'],
                      'labeled_rows': summary['labeled_rows'],
                      'note': 'Gói nội bộ, chưa ẩn danh để công khai.'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run_cli(main)
