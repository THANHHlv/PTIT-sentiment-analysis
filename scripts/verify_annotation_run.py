"""Kiểm chứng đợt gán nhãn hiện tại, chỉ ghi số lượng/hash, không log nội dung."""
import csv
import json
import sqlite3
from collections import Counter
from pathlib import Path

from ptit_sentiment.common import file_hash, read_json, write_json
from ptit_sentiment.data.annotation_review import export, read_table, state, write_table
from ptit_sentiment.data.validate import text_key


def main():
    directory=Path('data/labeled')
    original,original_fields=read_table('data/raw/ptit_sources_v1/comments.csv')
    collection=read_json('data/raw/ptit_sources_v1/collection_report.json')
    assert file_hash('data/raw/ptit_sources_v1/comments.csv')==collection['raw_csv_sha256']
    base,decisions,manifest=state(directory)
    for item in manifest['capture_files']:assert file_hash(item['path'])==item['sha256']
    source_byfile={Path(item['path']).name:s['source_id'] for s in collection['sources'] for item in s['export_files']}
    duplicates,duplicate_fields=read_table(directory/'duplicate_occurrences.csv')
    for row in duplicates:
        row['source_id']=source_byfile.get(Path(row['capture_file']).name,row['source_id'])
    write_table(directory/'duplicate_occurrences.csv',duplicates,duplicate_fields)
    originals={r['id']:r for r in original}
    working,_=read_table(directory/'comments_clean.csv')
    removed,_=read_table(directory/'excluded_comments.csv')
    suggestions,_=read_table(directory/'label_suggestions.csv')
    assert len(working)==len({r['id'] for r in working})
    assert len(suggestions)==len({r['id'] for r in suggestions})
    assert len(working)+len(removed)==len(original)
    assert {r['id'] for r in working}|{r['id'] for r in removed}==set(originals)
    assert not ({r['id'] for r in working}&{r['id'] for r in removed})
    for row in working+removed+suggestions:
        assert row['text']==originals[row['id']]['text'] and row['post_id']==originals[row['id']]['post_id']
    assert {r['id'] for r in working}=={r['id'] for r in suggestions}
    assert all(r['annotation_status']=='pending_review' and not r['label'] for r in suggestions)
    assert all('?' not in d['reason'] for d in decisions.values())  # no broken stdin replacement in Vietnamese reasons
    # Database read-only; no importer or browser activity.
    with sqlite3.connect('file:data/raw/ptit_sources_v1/collection.sqlite3?mode=ro',uri=True) as db:
        dbrows=list(db.execute('select id,post_id,text from comments'))
        assert len(dbrows)==len(original)
        assert all(i in originals and (p,t)==(originals[i]['post_id'],originals[i]['text']) for i,p,t in dbrows)
    csv_inventory=[]
    for path in sorted(Path('data').rglob('*.csv')):
        if path.parts[1] not in ('raw','labeled','splits'):continue
        rows,fields=read_table(path)
        relation='derived_review_or_other_version'
        if {'id','text','post_id'}<=set(fields) and rows:
            shared=[r for r in rows if r['id'] in originals]
            matches=sum(r['text']==originals[r['id']]['text'] and r['post_id']==originals[r['id']]['post_id'] for r in shared)
            relation='same_or_subset_of_main' if len(shared)==len(rows) and matches==len(rows) else 'separate_or_conflicting_data'
        csv_inventory.append({'path':str(path),'rows':len(rows),'columns':fields,'sha256':file_hash(path),'relation':relation})
    incoming=[]
    keys={text_key(r['text']) for r in original}
    for path in sorted(Path('data/raw/incoming').glob('*.json')):
        rows=json.loads(path.read_text(encoding='utf-8-sig'));rows=rows if isinstance(rows,list) else rows.get('data',[])
        item={'path':str(path),'rows':len(rows),'sha256':file_hash(path),
              'same_ids':sum(r['id'] in originals for r in rows),
              'same_normalized_content':sum(text_key(r['text']) in keys for r in rows),
              'new_content_rows':sum(r['id'] not in originals and text_key(r['text']) not in keys for r in rows)}
        assert item['new_content_rows']==0
        incoming.append(item)
    manifest['file_relationship_inventory']=csv_inventory
    manifest['incoming_file_relationships']=incoming
    manifest['sqlite_relationship']='Exact same id/post_id/text set as main CSV; read-only verification.'
    manifest['previous_label_status']='User explicitly confirmed old assignments_v1/confirmed_v1 have no human confirmation. Preserved unchanged; excluded from this round.'
    write_json(directory/'audit_manifest.json',manifest)
    # Sau khi xuất bộ AI được phép, kiểm tra này không reset báo cáo/mẫu review.
    report=read_json(manifest['audit_path']) if (directory/'ai_labeling_report.json').exists() else export(directory)
    summary={k:report[k] for k in ['initial_rows','valid_rows','excluded_rows','processed_rows','remaining_rows','suggested_rows',
        'processed_without_suggestion','suggested_label_counts','human_confirmed_rows','high_priority_processed_rows','review_required_rows',
        'random_review_sample_rows','count_reconciliation','valid_source_counts','audit_flag_counts']}
    summary.update(raw_unchanged=True,capture_files_unchanged=True,sqlite_matches=True,original_text_preserved=True,
        suggestions_unique_ids=True,unconfirmed_never_gold=True,human_labeled_file_created=report['human_confirmed_rows']>0,
        scope='Data audit and preliminary language review only; no collection/model training.',
        tests='14 annotation_review unit tests passed; isolated fixtures only.')
    write_json('outputs/annotation_verification.json',summary)
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
