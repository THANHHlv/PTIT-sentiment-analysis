"""Lưu các quyết định đã đọc trực tiếp, tuyệt đối không phân loại bằng từ khóa.

stdin: mỗi dòng seq|nhãn (P/N/U hoặc trống)|ưu tiên (L/M/H)|lý do.
N = negative, U = neutral. Thêm cột thứ năm exclude để loại mẫu đã đọc.
Lưu JSON quyết định có hash và gọi checkpoint record; không đụng nhãn chuẩn.
"""
import json
import sys
from pathlib import Path
from ptit_sentiment.data.annotation_review import record, state


def main():
    directory = Path(sys.argv[1])
    _, _, manifest = state(directory)
    rows = []
    for line in sys.stdin.read().splitlines():
        if not line.strip(): continue
        parts = line.split('|')
        if len(parts) not in (4,5): raise ValueError('Cần seq|label|priority|reason[|exclude].')
        seq, label, priority, reason = parts[:4]
        rows.append({'seq':int(seq),'suggested_label':{'P':'positive','N':'negative','U':'neutral','':''}[label],
                     'review_priority':{'L':'low','M':'medium','H':'high'}[priority],
                     'reason':reason,'action':parts[4] if len(parts)==5 else 'keep'})
    target = directory / 'next_language_batch.json'
    target.write_text(json.dumps({'base_sha256':manifest['base_sha256'],'decisions':rows},ensure_ascii=False),encoding='utf-8')
    report = record(directory,target)
    print({k:report[k] for k in ['processed_rows','remaining_rows','valid_rows','excluded_rows','suggested_label_counts']})


if __name__=='__main__': main()
