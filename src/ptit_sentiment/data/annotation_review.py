"""Kiểm toán, lưu đề xuất theo batch và ghép xác nhận độc lập của bốn người.

Không dự đoán bằng từ khóa, không gọi dịch vụ ngoài và không huấn luyện.
Các quyết định ngôn ngữ được nhập tường minh; nhãn chuẩn chỉ đến từ file người duyệt.
"""
import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path

import pandas as pd

from ptit_sentiment.common import file_hash, read_json, run_cli, write_json
from ptit_sentiment.data.validate import text_key
from ptit_sentiment.labels import LABELS

FIELDS = ['id', 'post_id', 'text', 'suggested_label', 'review_priority', 'reason',
          'label', 'annotation_status']
TASK_FIELDS = ['id', 'post_id', 'text', 'role', 'label', 'annotation_status',
               'reviewer', 'context_used', 'notes']


def read_table(path):
    """CSV UTF-8/BOM -> dict chuỗi; từ chối header trùng và dòng lệch cột."""
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError('Header CSV rỗng hoặc trùng.')
        rows = list(reader)
        if any(None in row or any(v is None for v in row.values()) for row in rows):
            raise ValueError('CSV có dòng thừa/thiếu ô.')
    return rows, reader.fieldnames


def write_table(path, rows, fields):
    """Ghi CSV BOM cho Windows, giữ text; thay tệp nguyên tử."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def flags(text):
    """Chỉ tìm dấu hiệu chất lượng cần duyệt, không suy ra cảm xúc."""
    found = []
    if '\ufffd' in text or any(ord(c) < 32 and c not in '\n\r\t' for c in text):
        found.append('unusual_characters')
    if re.search(r'(?:Xem thêm|See more|Thích\s+Trả lời|Like\s+Reply)', text, re.I):
        found.append('possible_ui_text')
    if text.rstrip().endswith(('...', '…')):
        found.append('possible_truncation_or_deliberate_ellipsis')
    if re.search(r'(?:tuyển dụng|tuyển sinh|sale|giảm giá|liên hệ|zalo|pass lại|cho thuê)', text, re.I):
        found.append('possible_advertisement_not_automatic_exclusion')
    words = text.strip().split()
    if (2 <= len(words) <= 5 and all(w[0:1].isupper() and w.isalpha() for w in words)) or re.fullmatch(r'(?:@[\w.]+\s*)+', text.strip()):
        found.append('possible_tag_or_name_only_not_automatic_exclusion')
    if text.lstrip().startswith(('=', '+', '-', '@')):
        found.append('excel_import_as_text_required')
    return found


def redact_preview(text):
    """Che URL, liên hệ và cụm viết hoa có thể là tên trong đầu ra đọc batch.

    Đây là bản xem riêng; không sửa text lưu trữ. Tên viết thường có thể chưa
    được nhận diện: không dùng bản xem làm công cụ xuất bản/ẩn danh tự động.
    """
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    text = re.sub(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', '[EMAIL]', text)
    text = re.sub(r'(?<!\w)(?:\+?84|0)[\d .-]{8,14}\d(?!\w)', '[SỐ LIÊN HỆ]', text)
    text = re.sub(r'@[\w.]+', '[TAG]', text)
    text = re.sub(r'Người tham gia ẩn danh\s*\d+', '[NGƯỜI ĐƯỢC NHẮC]', text, flags=re.I)
    text = re.sub(r'Anonymous participant\s*\d+|\b[A-Z][A-Za-z_]*\d{3,}\b', '[NGƯỜI ĐƯỢC NHẮC]', text, flags=re.I)
    words = text.split()
    i = 0
    while i < len(words):
        j = i
        while j < len(words) and words[j][0:1].isupper() and not words[j].isupper():
            j += 1
        if j - i >= 2:
            words[i:j] = ['[TÊN/CỤM VIẾT HOA]']
        i = max(i + 1, j if j == i else i + 1)
    return ' '.join(words)


def near_pairs(rows):
    """Ứng viên gần trùng: >=12 ký tự, Jaccard từ >=0.6, tỷ lệ ký tự >=0.9.

    Chỉ là phát hiện ứng viên; không tự xóa. Không bao quát diễn đạt lại ngữ nghĩa.
    """
    tokens, keys, inverted = {}, {}, defaultdict(set)
    for row in rows:
        key = text_key(row['text'])
        keys[row['id']] = key
        tokens[row['id']] = set(key.split())
        if len(key) >= 12:
            for token in tokens[row['id']]:
                inverted[token].add(row['id'])
    result, seen = [], set()
    for row in rows:
        a = row['id']; ka = keys[a]; ta = tokens[a]
        if len(ka) < 12:
            continue
        candidates = set().union(*(inverted[t] for t in ta)) if ta else set()
        for b in sorted(candidates):
            pair = tuple(sorted((a, b)))
            if a == b or pair in seen:
                continue
            seen.add(pair)
            kb = keys[b]; tb = tokens[b]
            if ka == kb or min(len(ka), len(kb)) / max(len(ka), len(kb)) < .8:
                continue
            if len(ta & tb) / len(ta | tb) < .6:
                continue
            ratio = SequenceMatcher(None, ka, kb, autojunk=False).ratio()
            if ratio >= .9:
                result.append({'id_a': a, 'id_b': b, 'similarity': round(ratio, 5)})
    return result


def audit(input_path, output_dir, capture_dir=None, source_report=None, audit_path=None):
    """Tạo bản làm việc mới, kiểm tra cấu trúc/nguồn/trùng; không sửa raw."""
    output = Path(output_dir)
    if any((output / name).exists() for name in ['audit_base.csv','audit_manifest.json','label_suggestions.csv','comments_clean.csv','annotation_batches']):
        raise ValueError('Thư mục làm việc đã có dữ liệu; dùng phiên bản mới hoặc resume bằng record.')
    rows, columns = read_table(input_path)
    if {'id', 'text'} - set(columns):
        raise ValueError('Cần cột id,text. post_id được để trống nhưng không được chế tạo.')
    if not rows:
        raise ValueError('Đầu vào không có dòng.')
    if source_report and read_json(source_report).get('raw_csv_sha256') != file_hash(input_path):
        raise ValueError('Hash bản chính không khớp collection_report.')
    seen, clean, excluded, conflicting_ids = {}, [], [], set()
    groups = defaultdict(list)
    for number, original in enumerate(rows, 2):
        row = dict(original)
        row.setdefault('post_id', '')
        reasons = []
        if not row['id'].strip():
            # Đây là mã nội bộ có định danh tách khỏi nguồn, không chế tạo post_id.
            row['source_comment_id'] = ''
            row['id'] = 'internal:' + hashlib.sha256(f'{file_hash(input_path)}:{number}'.encode()).hexdigest()[:20]
            row['id_origin'] = 'internal_missing_source_id'
        else:
            row['source_comment_id'] = row['id']
            row['id_origin'] = 'source'
        row['source_row'] = str(number)
        row['audit_flags'] = ';'.join(flags(row['text']))
        if not row['text'].strip(): reasons.append('empty_text')
        if row['id'] != row['id'].strip() or row['post_id'] != row['post_id'].strip():
            reasons.append('identifier_whitespace_requires_source_correction')
        if row['id'] in seen:
            reasons.append('duplicate_id' if (row['post_id'], row['text']) == seen[row['id']] else 'conflicting_id')
            if (row['post_id'], row['text']) != seen[row['id']]: conflicting_ids.add(row['id'])
        seen[row['id']] = (row['post_id'], row['text'])
        if re.fullmatch(r'(?:https?://\S+\s*)+', row['text'].strip()): reasons.append('url_only')
        if row['text'].strip() and re.fullmatch(r'[\s.,…!?_\-]+', row['text']): reasons.append('punctuation_only_no_sentiment_evidence')
        if reasons:
            excluded.append({**row, 'exclusion_reason': ';'.join(reasons), 'exclusion_stage': 'structural_audit'})
        else:
            clean.append(row)
            groups[text_key(row['text'])].append(row)
    if conflicting_ids:
        excluded.extend({**r, 'exclusion_reason':'conflicting_id', 'exclusion_stage':'structural_audit'}
                        for r in clean if r['id'] in conflicting_ids)
        clean = [r for r in clean if r['id'] not in conflicting_ids]
        groups = defaultdict(list)
        for row in clean: groups[text_key(row['text'])].append(row)
    # Giữ mọi id nguồn nếu trùng nội dung trong bản chính; yêu cầu xử lý trước split.
    duplicate_rows = []
    for key, members in groups.items():
        if len(members) > 1:
            for member in members:
                duplicate_rows.append({**member, 'canonical_id': members[0]['id'], 'relationship': 'same_normalized_content_in_main'})
    capture_inventory, capture_rows = [], []
    if capture_dir:
        source_files = {}
        if source_report:
            for source in read_json(source_report).get('sources', []):
                for item in source.get('export_files', []):
                    source_files[Path(item['path']).name] = source['source_id']
        for path in sorted(Path(capture_dir).glob('*.json')):
            data = json.loads(path.read_text(encoding='utf-8-sig'))
            data = data if isinstance(data, list) else data.get('data', data.get('comments', []))
            if not isinstance(data, list): raise ValueError(f'{path}: không có bảng bình luận.')
            capture_inventory.append({'path': str(path), 'rows': len(data), 'sha256': file_hash(path)})
            for record in data:
                record = {**record, 'capture_file': str(path), 'source_id':source_files.get(path.name, '')}
                capture_rows.append(record)
        main = {row['id']: row for row in rows}
        normalized = {text_key(row['text']): row['id'] for row in rows}
        for row in capture_rows:
            if row['id'] in main:
                if (row['post_id'], row['text']) != (main[row['id']]['post_id'], main[row['id']]['text']):
                    raise ValueError('Snapshot và CSV có id xung đột. Cần chọn/correct nguồn.')
            elif text_key(row['text']) in normalized:
                duplicate_rows.append({**row, 'canonical_id': normalized[text_key(row['text'])],
                                       'relationship': 'historical_content_dedup_other_source_id'})
            else:
                raise ValueError('Snapshot có nội dung mới ngoài bản chính. Cần người dùng chọn nguồn trước khi ghép.')
    pairs = near_pairs(clean)
    near_ids = {x[k] for x in pairs for k in ('id_a', 'id_b')}
    for row in clean:
        if row['id'] in near_ids:
            row['audit_flags'] = ';'.join(filter(None, [row['audit_flags'], 'near_duplicate_candidate']))
    fields = list(dict.fromkeys(columns + ['post_id', 'source_comment_id', 'id_origin', 'source_row', 'audit_flags']))
    output.mkdir(parents=True, exist_ok=True)
    write_table(output / 'audit_base.csv', clean, fields)
    write_table(output / 'audit_excluded.csv', excluded, fields + ['exclusion_reason', 'exclusion_stage'])
    write_table(output / 'duplicate_occurrences.csv', duplicate_rows,
                ['id', 'post_id', 'text', 'source_id', 'source_url', 'capture_file', 'canonical_id', 'relationship'])
    write_table(output / 'near_duplicates.csv', pairs, ['id_a', 'id_b', 'similarity'])
    report = {
        'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
        'input': str(input_path), 'input_sha256': file_hash(input_path), 'encoding': 'utf-8-sig (accepts plain UTF-8)',
        'columns': columns, 'identifier_types': 'strings', 'initial_rows': len(rows),
        'structurally_valid_rows': len(clean), 'structurally_excluded_rows': len(excluded),
        'duplicate_id_rows': len(rows) - len({r['id'] for r in rows}),
        'main_duplicate_content_groups': sum(len(v) > 1 for v in groups.values()),
        'historically_deduplicated_snapshot_rows': sum(r['relationship'].startswith('historical') for r in duplicate_rows),
        'capture_rows': len(capture_rows), 'capture_files': capture_inventory,
        'sources': dict(Counter(r.get('source_id', '') for r in rows)),
        'posts': len({r['post_id'] for r in rows if r.get('post_id', '').strip()}),
        'collection_timestamp_range': [min(r.get('collected_at','') for r in rows), max(r.get('collected_at','') for r in rows)],
        'missing_fields': {k: sum(not r.get(k, '').strip() for r in rows) for k in ['id', 'post_id', 'text', 'source_id', 'source_url']},
        'near_duplicate_pairs': len(pairs), 'audit_flag_counts': dict(Counter(f for r in clean for f in r['audit_flags'].split(';') if f)),
        'label_provenance': 'Raw has no human labels; previous assignments_v1/confirmed_v1 not human-confirmed per user.',
        'policy': 'Main CSV selected by exact snapshot/hash lineage; historical normalized content duplicates remain in duplicate_occurrences.csv; near duplicates retained for review.',
        'base_sha256': file_hash(output / 'audit_base.csv'), 'audit_path': str(audit_path or output / 'data_audit.json'),
    }
    write_json(output / 'audit_manifest.json', report)
    export(output)
    return report


def state(directory):
    """Đọc trạng thái batch, đối chiếu hash; mỗi id chỉ có một quyết định AI."""
    directory = Path(directory)
    manifest = read_json(directory / 'audit_manifest.json')
    if file_hash(directory / 'audit_base.csv') != manifest['base_sha256']:
        raise ValueError('Bản làm việc thay đổi; không resume trên dữ liệu khác.')
    base, _ = read_table(directory / 'audit_base.csv')
    decisions = {}
    for path in sorted((directory / 'suggestion_batches').glob('batch_*.json')):
        batch = read_json(path)
        if batch['base_sha256'] != manifest['base_sha256']:
            raise ValueError('Batch thuộc phiên bản khác.')
        for decision in batch['decisions']:
            if decision['id'] in decisions: raise ValueError('Một id có nhiều quyết định batch.')
            decisions[decision['id']] = decision
    return base, decisions, manifest


def record(directory, decisions_path):
    """Lưu batch các quyết định ngôn ngữ tường minh rồi tái xuất bảng/tổng tiến độ."""
    directory = Path(directory)
    base, old, manifest = state(directory)
    by_sequence = {i: row for i, row in enumerate(base, 1)}
    incoming = read_json(decisions_path)
    if isinstance(incoming, dict):
        # Hash bắt buộc trong file quyết định để tránh nhầm seq của phiên bản khác.
        if incoming.get('base_sha256') != manifest['base_sha256']:
            raise ValueError('File quyết định khác phiên bản bản làm việc.')
        incoming = incoming['decisions']
    if not isinstance(incoming, list) or not incoming:
        raise ValueError('Batch phải là danh sách không rỗng.')
    records, ids = [], set()
    for decision in incoming:
        row = by_sequence.get(decision.get('seq'))
        if not row: raise ValueError('seq ngoài bản làm việc.')
        if row['id'] in old or row['id'] in ids: raise ValueError('id đã xử lý, không ghi đè/lặp.')
        ids.add(row['id'])
        if decision.get('suggested_label', '') not in ('', *LABELS): raise ValueError('Nhãn đề xuất không hợp lệ.')
        if decision.get('review_priority') not in ('high', 'medium', 'low'): raise ValueError('Ưu tiên không hợp lệ.')
        if not decision.get('reason', '').strip(): raise ValueError('Thiếu lý do.')
        if decision.get('action', 'keep') not in ('keep', 'exclude'): raise ValueError('action không hợp lệ.')
        if not decision.get('suggested_label') and decision.get('action', 'keep') != 'exclude' and decision['review_priority'] != 'high':
            raise ValueError('Không đủ căn cứ phải high.')
        records.append({**decision, 'id': row['id']})
    target = directory / 'suggestion_batches'
    target.mkdir(exist_ok=True)
    index = len(list(target.glob('batch_*.json'))) + 1
    write_json(target / f'batch_{index:04d}.json', {'base_sha256': manifest['base_sha256'],
               'created_at': datetime.now(timezone.utc).isoformat(),
               'method': 'Direct language analysis by assistant; no keyword fallback or external API', 'decisions': records})
    return export(directory)


def export(directory, seed=42, sample_per_label=30):
    """Xuất tất cả dòng, review ưu tiên, mẫu ngẫu nhiên và checkpoint sau mỗi batch."""
    directory = Path(directory)
    base, decisions, manifest = state(directory)
    excluded, excluded_fields = read_table(directory / 'audit_excluded.csv')
    clean, suggestions = [], []
    for row in base:
        row['audit_flags'] = ';'.join(sorted(set(filter(None, row['audit_flags'].split(';'))) | set(flags(row['text']))))
        decision = decisions.get(row['id'])
        if decision and decision.get('action') == 'exclude':
            excluded.append({**row, 'exclusion_reason': decision['reason'], 'exclusion_stage': 'language_review'})
            continue
        clean.append(row)
        human = row.get('annotation_status') == 'human_confirmed' and row.get('label') in LABELS and bool(row.get('reviewer', '').strip())
        suggestions.append({**row, 'suggested_label': decision.get('suggested_label', '') if decision else '',
            'review_priority': decision['review_priority'] if decision else 'high',
            'reason': decision['reason'] if decision else 'Chưa được AI đọc; không có đề xuất.',
            'label': row['label'] if human else '', 'annotation_status': 'human_confirmed' if human else 'pending_review',
            'label_source': 'explicit_human_confirmation_in_input' if human else '',
            'ai_processed': 'yes' if decision else 'no',
            'context_used': 'comment_only_no_parent_or_post_text'})
    fields = list(base[0]) if base else ['id', 'post_id', 'text']
    write_table(directory / 'comments_clean.csv', clean, fields)
    out_fields = FIELDS + ['ai_processed', 'context_used', 'audit_flags', 'label_source']
    write_table(directory / 'label_suggestions.csv', suggestions, out_fields)
    required = [r for r in suggestions if r['review_priority'] == 'high']
    write_table(directory / 'review_required.csv', required, out_fields)
    unread = [r for r in suggestions if r['ai_processed'] == 'no']
    unread_path = directory / 'unprocessed_comments.csv'
    if unread:
        write_table(unread_path, unread, out_fields)
    elif unread_path.exists():
        unread_path.unlink()  # File sinh tự động rỗng, tiến độ vẫn nằm trong JSON.
    rng, samples = random.Random(seed), []
    for label in LABELS:
        candidates = [r for r in suggestions if r['suggested_label'] == label and r['review_priority'] != 'high']
        samples.extend(rng.sample(candidates, min(sample_per_label, len(candidates))))
    write_table(directory / 'review_sample.csv', samples, out_fields)
    write_table(directory / 'excluded_comments.csv', excluded, excluded_fields)
    summary = {**manifest, 'valid_rows': len(clean), 'excluded_rows': len(excluded),
        'valid_source_counts': dict(Counter(r.get('source_id','') for r in clean)),
        'audit_flag_counts': dict(Counter(f for r in base for f in r['audit_flags'].split(';') if f)),
        'exclusion_reasons': dict(Counter(r['exclusion_reason'] for r in excluded)),
        'processed_rows': len(decisions), 'remaining_rows': len(base) - len(decisions),
        'suggested_rows': sum(bool(r['suggested_label']) for r in suggestions),
        'processed_without_suggestion': sum(r['ai_processed'] == 'yes' and not r['suggested_label'] for r in suggestions),
        'review_required_rows': len(required), 'random_review_sample_rows': len(samples),
        'suggested_label_counts': {label: sum(r['suggested_label'] == label for r in suggestions) for label in LABELS},
        'high_priority_processed_rows': sum(r['review_priority']=='high' and r['ai_processed']=='yes' for r in suggestions),
        'human_confirmed_rows': sum(r['annotation_status']=='human_confirmed' for r in suggestions),
        'human_label_counts': {label: sum(r['label']==label and r['annotation_status']=='human_confirmed' for r in suggestions) for label in LABELS},
        'count_reconciliation': len(clean) + len(excluded) == manifest['initial_rows'],
        'progress_batch_count': len(list((directory / 'suggestion_batches').glob('batch_*.json')))}
    write_json(manifest['audit_path'], summary)
    write_json(directory / 'suggestion_progress.json', {k: summary[k] for k in ['processed_rows','remaining_rows','suggested_rows','processed_without_suggestion','progress_batch_count']})
    return summary


def prepare(directory, seed=42, common_count=40, output_dir=None):
    """Một primary/id; high và mẫu ngẫu nhiên có cross; mẫu chung cho cả bốn."""
    directory = Path(directory)
    rows, _ = read_table(directory / 'label_suggestions.csv')
    if common_count < 0: raise ValueError('common_count phải >=0.')
    target = Path(output_dir) if output_dir else directory / 'annotation_batches'
    if target.exists(): raise ValueError('Phân công đã tồn tại; không ghi đè bài người duyệt.')
    rng = random.Random(seed)
    shuffled = rows.copy(); rng.shuffle(shuffled)
    # Mẫu chung độc lập, không hiển thị suggested_label để giảm ảnh hưởng AI.
    common = {r['id'] for r in rng.sample(rows, min(common_count, len(rows)))}
    random_rows, _ = read_table(directory / 'review_sample.csv')
    cross = {r['id'] for r in rows if r['review_priority'] == 'high'} | {r['id'] for r in random_rows}
    tasks = {str(p): [] for p in range(1, 5)}; assignments = {}; common_rows = []
    for i, row in enumerate(shuffled):
        primary = i % 4 + 1
        people = list(range(1, 5)) if row['id'] in common else [primary]
        if row['id'] in cross and row['id'] not in common:
            people.append((primary - 1 + i % 3 + 1) % 4 + 1)
        assignments[row['id']] = []
        for p in people:
            role = 'primary' if p == primary else ('common' if row['id'] in common else 'cross_check')
            task = {**{k: row[k] for k in ['id','post_id','text']}, 'role': role, 'label': '',
                    'annotation_status': 'pending_review', 'reviewer': '', 'context_used': '', 'notes': ''}
            tasks[str(p)].append(task)
            assignments[row['id']].append({'person': str(p), 'role': role})
        if row['id'] in common: common_rows.append({k: row[k] for k in ['id','post_id','text']})
    target.mkdir(parents=True)
    write_table(target / 'base.csv', rows, FIELDS + ['ai_processed','context_used','audit_flags'])
    for person, records in tasks.items(): write_table(target / f'annotator_{person}.csv', records, TASK_FIELDS)
    write_table(target / 'common_sample.csv', common_rows, ['id','post_id','text'])
    manifest = {'schema_version': 1, 'base_sha256': file_hash(target / 'base.csv'),
        'data_sha256': file_hash(directory / 'comments_clean.csv'), 'seed': seed, 'samples': len(rows),
        'clean_file':str((directory / 'comments_clean.csv').resolve()),
        'common_samples': len(common), 'assignments': assignments,
        'task_counts': {p: len(v) for p,v in tasks.items()}, 'provenance': 'real',
        'confirmation_policy': 'Explicit human_confirmed + valid label + reviewer; all assigned reviewers required; no majority vote.'}
    write_json(target / 'assignment_manifest.json', manifest)
    return {k: manifest[k] for k in ['samples','common_samples','task_counts']}


def merge(assignment_dir, output_dir, resolutions=None):
    """Ghép nhãn đã xác nhận; pending/sai/mất dòng/bất đồng không vào labeled."""
    directory, output = Path(assignment_dir), Path(output_dir)
    if output.exists() and any(output.iterdir()): raise ValueError('Dùng thư mục output mới cho mỗi lần ghép.')
    manifest = read_json(directory / 'assignment_manifest.json')
    clean_file = manifest.get('clean_file', directory.parent / 'comments_clean.csv')
    if file_hash(clean_file) != manifest['data_sha256']:
        raise ValueError('Danh sách làm sạch đã đổi sau phân công; tạo đợt phân công phiên bản mới, giữ file cũ.')
    if file_hash(directory / 'base.csv') != manifest['base_sha256']: raise ValueError('base đã đổi.')
    base, _ = read_table(directory / 'base.csv')
    byid = {r['id']: r for r in base}
    records, issues = defaultdict(list), []
    for person in map(str, range(1,5)):
        path = directory / f'annotator_{person}.csv'
        expected = {k: next(a['role'] for a in v if a['person'] == person)
                    for k,v in manifest['assignments'].items() if any(a['person'] == person for a in v)}
        if not path.is_file():
            rows, fields = [], TASK_FIELDS
        else: rows, fields = read_table(path)
        if set(TASK_FIELDS) - set(fields): raise ValueError(f'{path.name}: thiếu cột.')
        if len({r['id'] for r in rows}) != len(rows): raise ValueError(f'{path.name}: id trùng.')
        if {r['id'] for r in rows} - set(expected): raise ValueError(f'{path.name}: id ngoài phân công.')
        for identifier in sorted(set(expected) - {r['id'] for r in rows}):
            issues.append({'id': identifier, 'person': person, 'issue': 'missing_row'})
        for row in rows:
            identifier = row['id']
            if any(row[k] != byid[identifier][k] for k in ['post_id','text']) or row['role'] != expected[identifier]:
                raise ValueError(f'{path.name}: sửa trường cố định của một id.')
            if row['label'] not in ('', *LABELS): issues.append({'id':identifier,'person':person,'issue':'invalid_label'})
            if row['annotation_status'] not in ('pending_review','human_confirmed'):
                issues.append({'id':identifier,'person':person,'issue':'invalid_status'})
            if row['annotation_status']=='human_confirmed' and not row['reviewer'].strip():
                issues.append({'id':identifier,'person':person,'issue':'confirmed_without_reviewer'})
            if row['annotation_status']=='human_confirmed' and not row['label']:
                issues.append({'id':identifier,'person':person,'issue':'confirmed_without_label'})
            valid = row['label'] in LABELS and row['annotation_status'] == 'human_confirmed' and bool(row['reviewer'].strip())
            records[identifier].append({**row, 'person': person, 'valid':valid})
    resolutions_byid = {}
    if resolutions:
        table, fields = read_table(resolutions)
        required = {'id','final_label','annotation_status','reviewer','reason'}
        if required - set(fields): raise ValueError('File phân xử thiếu cột.')
        if len({r['id'] for r in table}) != len(table) or {r['id'] for r in table} - set(byid): raise ValueError('Phân xử có id trùng/lạ.')
        resolutions_byid = {r['id']:r for r in table}
    confirmed, review, agreements = [], [], []
    for row in base:
        identifier = row['id']; votes = records[identifier]
        complete = len(votes) == len(manifest['assignments'][identifier]) and all(v['valid'] for v in votes)
        agreed = complete and len({v['label'] for v in votes}) == 1
        if complete and len(votes)>1:
            agreements.append({'id':identifier,'reviewers':len(votes),'unanimous':agreed})
        label = votes[0]['label'] if agreed else ''
        resolution = resolutions_byid.get(identifier)
        if resolution and not (complete and not agreed):
            raise ValueError('Chỉ phân xử id bất đồng đã có đủ xác nhận; không thay nhãn thiếu bằng quyết định tự động.')
        if resolution:
            if resolution['final_label'] not in LABELS or resolution['annotation_status'] != 'human_confirmed' or not resolution['reviewer'].strip() or not resolution['reason'].strip():
                raise ValueError('Quyết định phân xử cần nhãn hợp lệ, xác nhận, reviewer và reason.')
            label = resolution['final_label']
        if label:
            confirmed.append({**row,'label':label,'annotation_status':'human_confirmed',
                'label_source':'human_adjudication' if resolution else 'human_unanimous_confirmation',
                'reviewers':'|'.join(v['reviewer'] for v in votes)})
        else:
            review.append({**row,'issue_reason':'disagreement' if complete else 'missing_invalid_or_unconfirmed',
                'votes':'|'.join(f"member_{v['person']}:{v['label']}:{v['annotation_status']}" for v in votes),
                'final_label':'','annotation_status':'pending_review','reviewer':'','reason':''})
    output.mkdir(parents=True,exist_ok=True)
    write_table(output / 'needs_review.csv',review,['id','post_id','text','issue_reason','votes','final_label','annotation_status','reviewer','reason'])
    write_table(output / 'disagreements.csv',[r for r in review if r['issue_reason']=='disagreement'],['id','post_id','text','issue_reason','votes','final_label','annotation_status','reviewer','reason'])
    write_table(output / 'validation_issues.csv',issues,['id','person','issue'])
    confirmed_byid = {r['id']:r for r in confirmed}
    reviewed = [{**row, 'label':confirmed_byid[row['id']]['label'] if row['id'] in confirmed_byid else '',
                 'annotation_status':'human_confirmed' if row['id'] in confirmed_byid else 'pending_review'} for row in base]
    write_table(output / 'reviewed_comments.csv',reviewed,FIELDS+['ai_processed','context_used','audit_flags'])
    report = {'status':'confirmed' if confirmed else 'needs_human_review', 'provenance':manifest['provenance'],
        'samples':len(base),'confirmed_samples':len(confirmed),'pending_samples':len(review),
        'complete_dataset':not review and not issues, 'issues':issues,
        'common_or_cross_completed':len(agreements),'unanimous_agreement':sum(r['unanimous'] for r in agreements)/len(agreements) if agreements else None,
        'agreement_note':'Unanimity among complete independent assignments before adjudication; no fabricated agreement or kappa.',
        'label_counts':{label:sum(r['label']==label for r in confirmed) for label in LABELS},
        'assignment_manifest_sha256':file_hash(directory / 'assignment_manifest.json'),
        'annotator_file_hashes':{str(p):file_hash(directory / f'annotator_{p}.csv') if (directory / f'annotator_{p}.csv').exists() else None for p in range(1,5)},
        'resolutions_sha256':file_hash(resolutions) if resolutions else None,
        'confirmation_policy':manifest['confirmation_policy']}
    if confirmed:
        write_table(output / 'labeled_comments.csv',confirmed,['id','post_id','text','label','annotation_status','label_source','reviewers'])
        report['labeled_sha256']=file_hash(output / 'labeled_comments.csv')
    write_json(output / 'labeling_report.json',report)
    return report


def split_confirmed(input_path, output_dir, labeling_report, ratios=(.7,.15,.15), seed=42, attempts=2000):
    """Chia subset đã xác nhận qua bộ chia nhóm hiện có; không nhận pending/AI."""
    rows, fields = read_table(input_path)
    if 'annotation_status' not in fields or any(r['annotation_status']!='human_confirmed' for r in rows):
        raise ValueError('Chỉ chia mẫu human_confirmed; pending/AI bị từ chối.')
    if any(not r.get('post_id','').strip() for r in rows):
        raise ValueError('Thiếu post_id thật; bổ sung nguồn trước chia, không tạo post_id giả.')
    from ptit_sentiment.data.split import grouped_split, save_splits
    from ptit_sentiment.data.validate import read_csv
    frame = read_csv(input_path)
    planned = grouped_split(frame,ratios,seed,attempts)
    membership = {identifier:name for name,subset in planned.items() for identifier in subset['id']}
    crossing = [p for p in near_pairs(rows) if membership[p['id_a']] != membership[p['id_b']]]
    if crossing:
        raise ValueError(f'{len(crossing)} cặp gần trùng sẽ đi qua các tập; rà soát nội dung/loại bản dư trước khi chia. Chưa lưu bộ chia.')
    return save_splits(input_path,output_dir,ratios,seed,attempts,'real',labeling_report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    aud=commands.add_parser('audit'); aud.add_argument('--input',required=True); aud.add_argument('--output-dir',required=True)
    aud.add_argument('--capture-dir'); aud.add_argument('--source-report'); aud.add_argument('--audit-output')
    view=commands.add_parser('view'); view.add_argument('--directory',required=True); view.add_argument('--start',type=int,default=1); view.add_argument('--size',type=int,default=80)
    rec=commands.add_parser('record'); rec.add_argument('--directory',required=True); rec.add_argument('--decisions',required=True)
    exp=commands.add_parser('export'); exp.add_argument('--directory',required=True)
    prep=commands.add_parser('prepare'); prep.add_argument('--directory',required=True); prep.add_argument('--output-dir'); prep.add_argument('--seed',type=int,default=42); prep.add_argument('--common-count',type=int,default=40)
    mer=commands.add_parser('merge'); mer.add_argument('--assignment-dir',required=True); mer.add_argument('--output-dir',required=True); mer.add_argument('--resolutions')
    spl=commands.add_parser('split'); spl.add_argument('--input',required=True); spl.add_argument('--output-dir',required=True); spl.add_argument('--labeling-report',required=True)
    spl.add_argument('--seed',type=int,default=42); spl.add_argument('--attempts',type=int,default=2000)
    args=parser.parse_args()
    if args.command=='audit': report=audit(args.input,args.output_dir,args.capture_dir,args.source_report,args.audit_output)
    elif args.command=='view':
        base, decisions, _ = state(args.directory)
        for i,row in enumerate(base,1):
            if i >= args.start and i < args.start+args.size and row['id'] not in decisions:
                print(json.dumps({'seq':i,'text':redact_preview(row['text']),'flags':row['audit_flags']},ensure_ascii=False))
        return
    elif args.command=='record': report=record(args.directory,args.decisions)
    elif args.command=='export': report=export(args.directory)
    elif args.command=='prepare': report=prepare(args.directory,args.seed,args.common_count,args.output_dir)
    elif args.command=='merge': report=merge(args.assignment_dir,args.output_dir,args.resolutions)
    else: report=split_confirmed(args.input,args.output_dir,args.labeling_report,seed=args.seed,attempts=args.attempts)
    print(json.dumps({k:v for k,v in report.items() if k in ['initial_rows','valid_rows','excluded_rows','processed_rows','remaining_rows','suggested_rows','suggested_label_counts','status','confirmed_samples','pending_samples','samples','common_samples','task_counts']},ensure_ascii=False))


if __name__=='__main__': run_cli(main)
