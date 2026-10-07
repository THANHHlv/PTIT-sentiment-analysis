"""Xuất nhãn AI đã đọc tường minh và chia bài/nhóm gần trùng; không suy nhãn.

Chỉ dùng khi người dùng cho phép bộ nhãn AI ban đầu. Giữ dữ liệu nguồn,
checkpoint và nhãn người; không gọi API, học đặc trưng hoặc huấn luyện.
"""
import argparse
import hashlib
import random
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ptit_sentiment.common import file_hash, read_json, run_cli, write_json
from ptit_sentiment.data.annotation_review import read_table, state, write_table, near_pairs
from ptit_sentiment.data.split import check_disjoint, grouped_split
from ptit_sentiment.data.validate import statistics, validate_frame
from ptit_sentiment.labels import LABELS

ANNOTATION_FIELDS = ['id', 'post_id', 'text', 'label', 'label_source',
                     'annotation_status', 'reason', 'review_priority']
LIMITATION = ('Bộ nhãn AI ban đầu được người dùng cho phép. Test nhãn AI chỉ đo '
              'mức phù hợp với bộ nhãn đó, chưa thay thế nhãn người độc lập.')


def checked_origins(frame):
    """Từ chối pending/unresolved và metadata giả xác nhận trên nhãn AI."""
    for col in ('label_source', 'annotation_status'):
        if col not in frame:
            raise ValueError(f'Thiếu metadata {col}.')
    ai = frame['label_source'].eq('ai') & frame['annotation_status'].eq('ai_labeled')
    human = (frame['annotation_status'].eq('human_confirmed')
             & frame['label_source'].ne('ai') & frame['label_source'].str.strip().ne(''))
    if human.any() and ('reviewer' not in frame or frame.loc[human, 'reviewer'].str.strip().eq('').any()):
        raise ValueError('Nhãn người thiếu reviewer tường minh.')
    if not (ai | human).all():
        raise ValueError('Nguồn/trạng thái nhãn không hợp lệ; không nhận pending hoặc AI giả human_confirmed.')


def _archive_changed(path, history):
    """Giữ bản cũ theo hash trước khi cập nhật sản phẩm có thể tái tạo."""
    path = Path(path)
    if path.exists():
        dest = Path(history) / f'{path.stem}_{file_hash(path)[:12]}{path.suffix}'
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(path, dest)


def export_labels(directory):
    """Checkpoint hoàn chỉnh -> nhãn AI/human, unresolved và bảng đối soát từng ID."""
    directory = Path(directory)
    base, decisions, manifest = state(directory)
    if len(decisions) != len(base) or set(decisions) != {r['id'] for r in base}:
        raise ValueError('Chưa đọc hết dữ liệu hoặc có ID ngoài bản làm việc; tiếp tục batch trước.')
    if file_hash(manifest['input']) != manifest['input_sha256']:
        raise ValueError('Dữ liệu gốc khác phiên bản kiểm toán.')
    raw, _ = read_table(manifest['input'])
    exclusions, exclusion_fields = read_table(directory / 'audit_excluded.csv')
    labeled, unresolved, annotations = [], [], []
    for seq, source in enumerate(base, 1):
        decision = decisions[source['id']]
        if decision.get('seq') != seq or not decision.get('reason', '').strip():
            raise ValueError('Checkpoint có thứ tự/lý do không hợp lệ.')
        label = decision.get('suggested_label', '')
        if label not in ('', *LABELS) or decision.get('review_priority') not in ('high', 'medium', 'low'):
            raise ValueError('Checkpoint có nhãn/ưu tiên không hợp lệ.')
        human = source.get('annotation_status') == 'human_confirmed'
        if human and (source.get('label') not in LABELS or not source.get('reviewer', '').strip()):
            raise ValueError('Nhãn người trong đầu vào chưa đủ xác nhận.')
        if decision.get('action', 'keep') == 'exclude':
            if human:
                raise ValueError('Không tự loại dòng có nhãn người xác nhận.')
            exclusions.append({**source, 'exclusion_reason': decision['reason'], 'exclusion_stage': 'language_review'})
            continue
        row = {**source, 'label': source['label'] if human else label,
               'label_source': (source.get('label_source') or 'explicit_human_confirmation_in_input') if human else 'ai',
               'annotation_status': 'human_confirmed' if human else ('ai_labeled' if label else 'unresolved'),
               'reason': decision['reason'], 'review_priority': decision['review_priority'],
               'context_used': 'comment_only_no_parent_or_post_text'}
        annotations.append(row)
        (labeled if row['label'] else unresolved).append(row)
    fields = ANNOTATION_FIELDS + [k for k in list(base[0]) + ['context_used'] if k not in ANNOTATION_FIELDS]
    frame = pd.DataFrame(labeled, columns=fields)
    if labeled:
        validate_frame(frame)
        checked_origins(frame)
    lookup = {r['id']: r for r in raw}
    combined = labeled + unresolved + exclusions
    counts = Counter(r['id'] for r in combined)
    if set(counts) != set(lookup) or any(v != 1 for v in counts.values()) or len(raw) != len(lookup):
        raise ValueError('Đối soát ID thất bại; không xuất kết quả mất/nhân mẫu.')
    if any((r['post_id'], r['text']) != (lookup[r['id']]['post_id'], lookup[r['id']]['text']) for r in combined):
        raise ValueError('Text hoặc post_id khác dữ liệu gốc.')
    for name, rows, columns in [('labeled_comments.csv', labeled, fields), ('unresolved.csv', unresolved, fields),
                                ('excluded_comments.csv', exclusions, exclusion_fields)]:
        _archive_changed(directory / name, directory / 'history/ai_exports')
        write_table(directory / name, rows, columns)
    rng = random.Random(42)
    sample = [{**r, 'sample_role': 'high_priority'} for r in annotations if r['review_priority'] == 'high']
    for label in LABELS:
        candidates = [r for r in labeled if r['label'] == label and r['review_priority'] != 'high']
        sample += [{**r, 'sample_role': 'stratified_random'} for r in rng.sample(candidates, min(30, len(candidates)))]
    _archive_changed(directory / 'review_sample.csv', directory / 'history/ai_exports')
    write_table(directory / 'review_sample.csv', sample, fields + ['sample_role'])
    reconciliation = [{'id': r['id'], 'destination': target, 'source_row': r.get('source_row', '')}
                      for target, rows in [('labeled', labeled), ('unresolved', unresolved), ('excluded', exclusions)] for r in rows]
    write_table(directory / 'id_reconciliation.csv', reconciliation, ['id', 'destination', 'source_row'])
    report = {'schema_version': 1, 'status': 'ai_initial', 'provenance': 'real',
              'created_at': datetime.now(timezone.utc).isoformat(), 'input_sha256': manifest['input_sha256'],
              'labeled_sha256': file_hash(directory / 'labeled_comments.csv'),
              'base_sha256': manifest['base_sha256'], 'initial_rows': len(raw), 'processed_rows': len(decisions),
              'remaining_rows': 0, 'labeled_rows': len(labeled), 'unresolved_rows': len(unresolved),
              'excluded_rows': len(exclusions), 'label_counts': statistics(frame)['label_counts'],
              'label_source_counts': dict(Counter(r['label_source'] for r in labeled)),
              'annotation_status_counts': dict(Counter(r['annotation_status'] for r in labeled)),
              'human_labels_confirmed': bool(labeled) and not frame['label_source'].eq('ai').any(),
              'human_confirmed_rows': sum(r['annotation_status'] == 'human_confirmed' for r in labeled),
              'method': 'Direct language analysis of each comment; explicit decisions; no keyword fallback/API.',
              'context': 'Only actual comment text. No parent/post text was available or inferred.',
              'limitation': LIMITATION, 'count_reconciliation': True,
              'batch_hashes': {p.name: file_hash(p) for p in sorted((directory / 'suggestion_batches').glob('batch_*.json'))}}
    write_json(directory / 'ai_labeling_report.json', report)
    audit = read_json(manifest['audit_path'])
    audit.update({**{k: report[k] for k in ['labeled_rows', 'unresolved_rows', 'excluded_rows', 'remaining_rows',
                                          'label_counts', 'label_source_counts', 'annotation_status_counts']},
                  'labeling_policy': 'ai_initial_authorized_by_user', 'ai_dataset': report})
    write_json(manifest['audit_path'], audit)
    return labeled, report


def connected_posts(rows, all_rows, pairs, historical_duplicates):
    """Liên kết post_id thật qua gần trùng/trùng lịch sử; không tạo post_id giả."""
    if any(not r['post_id'].strip() for r in rows):
        raise ValueError('Thiếu post_id thật; không thể chia theo bài trong quy trình này.')
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a, b):
        if a and b:
            a, b = find(a), find(b)
            parent[max(a, b)] = min(a, b)
    by_id = {r['id']: r for r in all_rows}
    for row in rows:
        find(row['post_id'])
    for pair in pairs:
        if pair['id_a'] not in by_id or pair['id_b'] not in by_id:
            raise ValueError('Bảng gần trùng có ID ngoài nguồn.')
        union(by_id[pair['id_a']]['post_id'], by_id[pair['id_b']]['post_id'])
    for row in historical_duplicates:
        if row['canonical_id'] not in by_id:
            raise ValueError('Quan hệ trùng lịch sử có ID đại diện ngoài nguồn.')
        union(row['post_id'], by_id[row['canonical_id']]['post_id'])
    # Mã này chỉ là nhóm chia nội bộ, không dùng thay post_id trong CSV sản phẩm.
    return {r['post_id']: 'split_component_' + hashlib.sha256(find(r['post_id']).encode()).hexdigest()[:16] for r in rows}


def split_ai_dataset(directory, output_dir, seed=42, attempts=10000):
    """Xuất version mới, giữ nguồn nhãn và toàn bộ liên kết bài/gần trùng."""
    directory, output_dir = Path(directory), Path(output_dir)
    rows, report = export_labels(directory)
    frame = pd.DataFrame(rows)
    _, _, audit_manifest = state(directory)
    raw, _ = read_table(audit_manifest['input'])
    pairs, _ = read_table(directory / 'near_duplicates.csv')
    historical, _ = read_table(directory / 'duplicate_occurrences.csv')
    extra = near_pairs(rows)
    pair_map = {tuple(sorted((p['id_a'], p['id_b']))): p for p in pairs + extra}
    pairs = list(pair_map.values())
    post_groups = connected_posts(rows, raw, pairs, historical)
    temporary = frame.copy()
    temporary['post_id'] = temporary['post_id'].map(post_groups)
    frames = grouped_split(temporary, (0.7, 0.15, 0.15), seed, attempts)
    originals = frame.set_index('id')['post_id']
    for subset in frames.values():
        subset['post_id'] = subset['id'].map(originals)
    check_disjoint(frames)
    memberships = {r['id']: name for name, sub in frames.items() for r in sub.to_dict('records')}
    if len(memberships) != len(rows) or set(memberships) != {r['id'] for r in rows}:
        raise ValueError('Mất/nhân ID khi chia.')
    crossing = [p for p in pairs if p['id_a'] in memberships and p['id_b'] in memberships
                and memberships[p['id_a']] != memberships[p['id_b']]]
    components = {}
    for name, subset in frames.items():
        for post in subset['post_id']:
            component = post_groups[post]
            if component in components and components[component] != name:
                raise ValueError('Nhóm trùng/gần trùng bị chia qua nhiều tập.')
            components[component] = name
    if crossing:
        raise ValueError('Gần trùng giao nhau giữa các tập.')
    version = 'ai_initial_' + report['labeled_sha256'][:12] + f'_seed{seed}'
    # Chỉ đụng các file đích xác định; các thư mục bộ chia cũ giữ nguyên.
    targets = ['train.csv', 'validation.csv', 'test.csv', 'manifest.json', 'metadata.json',
               'labeling_report.json', 'split_groups.csv', 'near_duplicate_constraints.csv']
    if any((output_dir / name).exists() for name in targets):
        raise ValueError('Bộ chia đích đã tồn tại; chọn --output-dir mới, không ghi đè.')
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'schema_version': 2, 'source': str((directory / 'labeled_comments.csv').resolve()),
                'source_sha256': report['labeled_sha256'], 'provenance': 'real', 'dataset_version': version,
                'labeling_policy': 'ai_initial', 'human_labels_confirmed': report['human_labels_confirmed'],
                'method': 'seeded_group_search_post_id_and_duplicate_components_v1', 'seed': seed,
                'requested_ratios': [0.7, 0.15, 0.15], 'attempts': attempts,
                'ratio_note': 'Tỷ lệ gần đúng vì nhóm bài không chia nhỏ; không ép cân bằng nhãn.',
                'label_counts': report['label_counts'], 'label_source_counts': report['label_source_counts'],
                'annotation_status_counts': report['annotation_status_counts'], 'limitation': LIMITATION,
                'post_groups': len(post_groups), 'split_components': len(set(post_groups.values())),
                'historical_duplicate_constraints': len(historical), 'known_near_duplicate_pairs': len(pairs),
                'checks': {'all_ids_partitioned_once': True, 'id_intersections': 0, 'post_intersections': 0,
                           'normalized_content_intersections': 0, 'known_near_duplicate_intersections': 0,
                           'duplicate_component_intersections': 0, 'all_three_labels_each_split': True}, 'splits': {}}
    for name, subset in frames.items():
        checked_origins(subset)
        path = output_dir / f'{name}.csv'
        write_table(path, subset.to_dict('records'), list(frame.columns))
        manifest['splits'][name] = {**statistics(subset), 'actual_ratio': len(subset) / len(frame),
                                    'label_source_counts': dict(Counter(subset['label_source'])),
                                    'annotation_status_counts': dict(Counter(subset['annotation_status'])),
                                    'sha256': file_hash(path), 'ids': subset['id'].tolist(),
                                    'post_ids': sorted(set(subset['post_id']))}
    write_json(output_dir / 'labeling_report.json', report)
    manifest['labeling_report_sha256'] = file_hash(output_dir / 'labeling_report.json')
    write_table(output_dir / 'split_groups.csv', [{'id': r['id'], 'post_id': r['post_id'],
                  'split_component': post_groups[r['post_id']], 'split': memberships[r['id']]} for r in rows],
                  ['id', 'post_id', 'split_component', 'split'])
    write_table(output_dir / 'near_duplicate_constraints.csv', pairs, ['id_a', 'id_b', 'similarity'])
    manifest['split_groups_sha256'] = file_hash(output_dir / 'split_groups.csv')
    manifest['near_duplicate_constraints_sha256'] = file_hash(output_dir / 'near_duplicate_constraints.csv')
    write_json(output_dir / 'manifest.json', manifest)
    write_json(output_dir / 'metadata.json', manifest)
    # Một bộ chia trên mỗi thư mục đích đã được bảo vệ không ghi đè.
    # Phiên bản nằm trong manifest; không nhân bản toàn bộ CSV dưới versions/.
    audit = read_json(audit_manifest['audit_path'])
    audit['ai_splits'] = {'directory': str(output_dir), 'dataset_version': version,
                         'manifest_sha256': file_hash(output_dir / 'manifest.json'),
                         'checks': manifest['checks'], 'splits': {n: {k: v for k, v in s.items() if k not in ('ids', 'post_ids')}
                         for n, s in manifest['splits'].items()}}
    write_json(audit_manifest['audit_path'], audit)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', default='data/labeled')
    parser.add_argument('--output-dir', default='data/splits')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--attempts', type=int, default=10000)
    args = parser.parse_args()
    manifest = split_ai_dataset(args.directory, args.output_dir, args.seed, args.attempts)
    print({n: {k: s[k] for k in ('rows', 'posts', 'label_counts', 'actual_ratio')}
           for n, s in manifest['splits'].items()})


if __name__ == '__main__':
    run_cli(main)
