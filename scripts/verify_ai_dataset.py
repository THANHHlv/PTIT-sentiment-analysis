"""Đối soát bộ AI thật và bộ chia hiện tại, chỉ đọc dữ liệu; không thu thập/train."""
import argparse
from collections import Counter
from pathlib import Path
import sqlite3
import pandas as pd

from ptit_sentiment.common import file_hash, read_json, write_json
from ptit_sentiment.data.ai_dataset import checked_origins
from ptit_sentiment.data.annotation_review import read_table, state
from ptit_sentiment.data.split import load_split_set
from ptit_sentiment.data.validate import read_csv


def verify(directory, split_dir, output):
    """Đối chiếu mọi ID/text/source với raw và batch; lưu thống kê không chứa nội dung."""
    directory, split_dir = Path(directory), Path(split_dir)
    metadata = read_json(split_dir / 'manifest.json')
    if metadata.get('storage_layout') == 'compact_v1':
        return verify_compact(directory, split_dir, output)
    base, decisions, manifest = state(directory)
    raw, _ = read_table(manifest['input'])
    assert file_hash(manifest['input']) == manifest['input_sha256'], 'Raw bị đổi'
    assert len(base) == len(decisions), 'Checkpoint chưa hoàn chỉnh'
    for info in manifest['capture_files']:
        assert file_hash(info['path']) == info['sha256'], 'Snapshot nguồn bị đổi'
    by_id = {r['id']: r for r in raw}
    tables = {name: read_table(directory / filename)[0] for name, filename in [
        ('labeled', 'labeled_comments.csv'), ('unresolved', 'unresolved.csv'), ('excluded', 'excluded_comments.csv')]}
    combined = [r for rows in tables.values() for r in rows]
    ids = Counter(r['id'] for r in combined)
    assert len(raw) == len(by_id) == len(combined)
    assert set(ids) == set(by_id) and all(v == 1 for v in ids.values()), 'Mất/nhân ID'
    for row in combined:
        assert all(row[k] == by_id[row['id']][k] for k in ['post_id', 'text', 'source_id', 'source_url', 'collected_at']), 'Nguồn/text bị đổi'
    frame = read_csv(directory / 'labeled_comments.csv')
    checked_origins(frame)
    assert all(not r['label'] and r['annotation_status'] == 'unresolved' for r in tables['unresolved'])
    for row in tables['labeled']:
        if row['label_source'] == 'ai':
            assert row['label'] == decisions[row['id']]['suggested_label'], 'Nhãn khác quyết định đã đọc'
        assert row['reason'] == decisions[row['id']]['reason']
    frames, metadata, split_hash = load_split_set(split_dir)
    assert file_hash(directory / 'labeled_comments.csv') == metadata['source_sha256']
    original_labels = {r['id']: r for r in tables['labeled']}
    split_rows = [r for subset in frames.values() for r in subset.to_dict('records')]
    assert len(split_rows) == len(original_labels) and {r['id'] for r in split_rows} == set(original_labels)
    assert all(r == original_labels[r['id']] for r in split_rows), 'Bộ chia sửa nhãn hoặc metadata'
    samples, _ = read_table(directory / 'review_sample.csv')
    assert len(samples) == len({r['id'] for r in samples})
    assert set(r['id'] for r in samples) <= {r['id'] for r in tables['labeled'] + tables['unresolved']}
    random_counts = dict(Counter(r['label'] for r in samples if r['sample_role'] == 'stratified_random'))
    assert random_counts == {'positive': 30, 'neutral': 30, 'negative': 30}
    db_path = Path(manifest['input']).parent / 'collection.sqlite3'
    with sqlite3.connect(db_path.resolve().as_uri() + '?mode=ro', uri=True) as db:
        database = list(db.execute('select id,post_id,text from comments'))
    assert len(database) == len(raw)
    assert all(i in by_id and (p, t) == (by_id[i]['post_id'], by_id[i]['text']) for i, p, t in database)
    report = {'raw_rows': len(raw), 'labeled_rows': len(frame), 'unresolved_rows': len(tables['unresolved']),
              'excluded_rows': len(tables['excluded']), 'remaining_rows': len(base) - len(decisions),
              'raw_and_captures_unchanged': True, 'sqlite_matches_raw': True, 'original_text_and_source_preserved': True,
              'every_input_id_accounted_once': True, 'ai_never_human_confirmed': True,
              'all_split_rows_equal_labeled_source': True, 'review_sample_random_counts': random_counts,
              'split_checks': metadata['checks'], 'dataset_version': metadata['dataset_version'],
              'split_manifest_sha256': split_hash,
              'files': {str(directory / f): file_hash(directory / f) for f in ['labeled_comments.csv', 'unresolved.csv',
                       'excluded_comments.csv', 'review_sample.csv', 'ai_labeling_report.json']},
              'scope': 'AI initial labeling and splitting only; no collection/vectorizer fitting/model training.'}
    write_json(output, report)
    audit = read_json(manifest['audit_path'])
    audit['final_verification'] = report
    audit['review_sample_rows'] = len(samples)
    audit['review_sample_high_priority_rows'] = sum(r['sample_role'] == 'high_priority' for r in samples)
    write_json(manifest['audit_path'], audit)
    print({k: report[k] for k in ['raw_rows', 'labeled_rows', 'unresolved_rows', 'excluded_rows', 'remaining_rows', 'split_checks']})
    return report


def verify_compact(directory, split_dir, output):
    """Kiểm tra bộ tối giản; nguồn thô đã được lưu riêng, không giả kiểm tra live."""
    source = read_csv(directory / 'labeled_comments.csv')
    checked_origins(source)
    frames, metadata, digest = load_split_set(split_dir)
    assert file_hash(directory / 'labeled_comments.csv') == metadata['source_sha256']
    joined = pd.concat(list(frames.values()), ignore_index=True)
    assert set(joined.columns) == set(source.columns)
    assert source.sort_values('id').reset_index(drop=True).equals(
        joined[list(source.columns)].sort_values('id').reset_index(drop=True))
    audit = metadata['embedded_metadata']['labeling_report']
    assert len(source) == audit['labeled_rows']
    assert dict(Counter(source['label'])) == audit['label_counts']
    report = {'storage_layout': 'compact_v1', 'labeled_rows': len(source),
              'source_sha256': metadata['source_sha256'], 'split_manifest_sha256': digest,
              'label_counts': dict(Counter(source['label'])),
              'label_source_counts': dict(Counter(source['label_source'])),
              'splits': {n: len(f) for n, f in frames.items()}, 'split_checks': metadata['checks'],
              'all_split_rows_equal_labeled_source': True, 'ai_never_human_confirmed': True,
              'source_audit_note': 'Raw/checkpoint/unresolved/excluded archived outside data; original audit in outputs/data_audit.json.',
              'scope': 'Validate current labeled data and splits only; no collection or training.'}
    write_json(output, report)
    print({k: report[k] for k in ['storage_layout', 'labeled_rows', 'label_counts', 'splits', 'split_checks']})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', default='data/labeled')
    parser.add_argument('--split-dir', default='data/splits')
    parser.add_argument('--output', default='outputs/ai_dataset_verification.json')
    args = parser.parse_args()
    verify(args.directory, args.split_dir, args.output)
