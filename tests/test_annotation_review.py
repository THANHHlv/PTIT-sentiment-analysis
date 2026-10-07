"""Fixture tạm cho workflow kiểm toán/xác nhận; không thu thập hoặc train."""
import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from ptit_sentiment.common import file_hash, read_json, write_json
from ptit_sentiment.data.annotation_review import (FIELDS, TASK_FIELDS, audit, flags, merge,
    prepare, read_table, record, split_confirmed, state, write_table)
from ptit_sentiment.data.split import load_split_set


class AnnotationReviewTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'raw.csv'
        self.work = self.root/'review'
        self.rows = [{'id':'001','post_id':'010','text':'Fixture Alpha'},
                     {'id':'002','post_id':'011','text':'Fixture Bravo'},
                     {'id':'003','post_id':'012','text':'Fixture Charlie'}]
        write_table(self.source,self.rows,['id','post_id','text'])

    def setup_work(self, rows=None):
        if rows is not None: write_table(self.source,rows,list(rows[0]))
        before=file_hash(self.source)
        audit(self.source,self.work)
        self.assertEqual(before,file_hash(self.source))
        return self.work

    def confirm(self, work, skip=None):
        assignments=work/'annotation_batches'
        for person in range(1,5):
            path=assignments/f'annotator_{person}.csv'
            rows,_=read_table(path)
            for row in rows:
                if row['id']==skip: continue
                row.update(label={'001':'positive','002':'neutral','003':'negative'}[row['id']],
                           annotation_status='human_confirmed',reviewer=f'fixture_reviewer_{person}')
            write_table(path,rows,TASK_FIELDS)
        return assignments

    def test_audit_preserves_text_ids_and_handles_empty_internal_ids(self):
        rows=[{'id':'0001','post_id':'0010','text':'😢'},
              {'id':'','post_id':'','text':'không tốt'},
              {'id':'003','post_id':'012','text':'...'},
              {'id':'004','post_id':'013','text':'https://example.invalid'},
              {'id':'005','post_id':'014','text':'không tệ'}]
        work=self.setup_work(rows)
        clean,_=read_table(work/'comments_clean.csv')
        self.assertEqual(clean[0]['id'],'0001')
        self.assertEqual(clean[0]['text'],'😢')
        self.assertEqual(clean[1]['post_id'],'')
        self.assertTrue(clean[1]['id'].startswith('internal:'))
        excluded,_=read_table(work/'excluded_comments.csv')
        self.assertEqual(len(clean)+len(excluded),5)

    def test_conflicting_ids_are_both_quarantined(self):
        work=self.setup_work([*self.rows,{'id':'001','post_id':'010','text':'Changed fixture'}])
        clean,_=read_table(work/'comments_clean.csv')
        self.assertNotIn('001',{r['id'] for r in clean})
        excluded,_=read_table(work/'excluded_comments.csv')
        self.assertEqual(sum(r['id']=='001' for r in excluded),2)

    def test_resume_unique_ids_hash_guard_no_keyword_labeling(self):
        work=self.setup_work()
        packet=self.root/'packet.json'
        write_json(packet,{'base_sha256':state(work)[2]['base_sha256'],'decisions':[
            {'seq':1,'suggested_label':'positive','review_priority':'low','reason':'Explicit fixture decision'}]})
        record(work,packet)
        with self.assertRaisesRegex(ValueError,'đã xử lý'):record(work,packet)
        suggestions,_=read_table(work/'label_suggestions.csv')
        self.assertEqual(suggestions[0]['label'],'')
        self.assertEqual(suggestions[1]['suggested_label'],'')
        self.assertEqual(suggestions[1]['ai_processed'],'no')
        self.assertEqual(suggestions[0]['annotation_status'],'pending_review')
        write_json(packet,{'base_sha256':'wrong','decisions':[]})
        with self.assertRaisesRegex(ValueError,'phiên bản'):record(work,packet)

    def test_input_human_confirmation_preserved(self):
        rows=[{**r,'label':'positive','annotation_status':'human_confirmed','reviewer':'fixture human'} for r in self.rows]
        work=self.setup_work(rows)
        suggestions,_=read_table(work/'label_suggestions.csv')
        self.assertTrue(all(r['label']=='positive' and r['annotation_status']=='human_confirmed' for r in suggestions))

    def test_common_assignments_have_four_independent_rows(self):
        work=self.setup_work(); report=prepare(work,common_count=3)
        self.assertEqual(report['task_counts'],{str(p):3 for p in range(1,5)})
        for p in range(1,5):
            rows,fields=read_table(work/'annotation_batches'/f'annotator_{p}.csv')
            self.assertNotIn('suggested_label',fields)
            self.assertTrue(all(r['label']=='' and r['annotation_status']=='pending_review' for r in rows))

    def test_copied_suggestion_without_confirmation_never_gold(self):
        work=self.setup_work();prepare(work,common_count=3)
        for p in range(1,5):
            path=work/'annotation_batches'/f'annotator_{p}.csv';rows,_=read_table(path)
            for row in rows:row.update(label='positive',reviewer='fixture reviewer')
            write_table(path,rows,TASK_FIELDS)
        report=merge(work/'annotation_batches',self.root/'merged')
        self.assertEqual(report['confirmed_samples'],0)
        self.assertFalse((self.root/'merged/labeled_comments.csv').exists())

    def test_partial_human_labels_only_and_missing_rows_reported(self):
        work=self.setup_work();prepare(work,common_count=3)
        directory=self.confirm(work,skip='003')
        path=directory/'annotator_1.csv';rows,_=read_table(path)
        write_table(path,[r for r in rows if r['id']!='003'],TASK_FIELDS)
        report=merge(directory,self.root/'merged')
        self.assertEqual(report['confirmed_samples'],2)
        self.assertEqual(report['pending_samples'],1)
        self.assertTrue(any(r['issue']=='missing_row' for r in report['issues']))
        labeled,_=read_table(self.root/'merged/labeled_comments.csv')
        self.assertEqual({r['id'] for r in labeled},{'001','002'})

    def test_no_majority_vote_and_explicit_resolution(self):
        work=self.setup_work();prepare(work,common_count=3);directory=self.confirm(work)
        path=directory/'annotator_4.csv';rows,_=read_table(path)
        next(r for r in rows if r['id']=='001')['label']='negative'
        write_table(path,rows,TASK_FIELDS)
        report=merge(directory,self.root/'merged')
        self.assertEqual(report['confirmed_samples'],2)
        disagreements,_=read_table(self.root/'merged/disagreements.csv')
        self.assertEqual([r['id'] for r in disagreements],['001'])
        decision={'id':'001','final_label':'neutral','annotation_status':'human_confirmed','reviewer':'fixture adjudicator','reason':'Fixture-only arbitration'}
        resolutions=self.root/'resolution.csv';write_table(resolutions,[decision],list(decision))
        result=merge(directory,self.root/'resolved',resolutions)
        self.assertEqual(result['confirmed_samples'],3)
        self.assertEqual(result['unanimous_agreement'],2/3)

    def test_invalid_labels_and_duplicate_unknown_ids(self):
        work=self.setup_work();prepare(work,common_count=3)
        path=work/'annotation_batches/annotator_1.csv';rows,_=read_table(path)
        rows[0]['label']='positve';write_table(path,rows,TASK_FIELDS)
        report=merge(work/'annotation_batches',self.root/'invalid')
        self.assertTrue(any(r['issue']=='invalid_label' for r in report['issues']))
        write_table(path,[*rows,rows[0]],TASK_FIELDS)
        with self.assertRaisesRegex(ValueError,'trùng'):merge(work/'annotation_batches',self.root/'dup')
        rows[0]['id']='unknown';write_table(path,rows,TASK_FIELDS)
        with self.assertRaisesRegex(ValueError,'ngoài'):merge(work/'annotation_batches',self.root/'unknown')

    def test_immutable_text_and_pending_cannot_be_adjudicated(self):
        work=self.setup_work();prepare(work,common_count=3)
        resolutions=self.root/'resolution.csv'
        row={'id':'001','final_label':'positive','annotation_status':'human_confirmed','reviewer':'fixture','reason':'fixture'}
        write_table(resolutions,[row],list(row))
        with self.assertRaisesRegex(ValueError,'Chỉ phân xử'):merge(work/'annotation_batches',self.root/'bad',resolutions)
        path=work/'annotation_batches/annotator_1.csv';rows,_=read_table(path);rows[0]['text']='tampered'
        write_table(path,rows,TASK_FIELDS)
        with self.assertRaisesRegex(ValueError,'cố định'):merge(work/'annotation_batches',self.root/'tampered')

    def test_split_rejects_pending_missing_posts_and_inadequate_groups(self):
        rows=[{**r,'label':'positive','annotation_status':'pending_review'} for r in self.rows]
        write_table(self.source,rows,list(rows[0]))
        with self.assertRaisesRegex(ValueError,'human_confirmed'):split_confirmed(self.source,self.root/'split',self.root/'report.json')
        for row in rows:row.update(annotation_status='human_confirmed',post_id='')
        write_table(self.source,rows,list(rows[0]))
        with self.assertRaisesRegex(ValueError,'post_id'):split_confirmed(self.source,self.root/'split',self.root/'report.json')
        for row,original in zip(rows,self.rows):row['post_id']=original['post_id']
        write_table(self.source,rows,list(rows[0]))
        with self.assertRaisesRegex(ValueError,'quá nhỏ'):split_confirmed(self.source,self.root/'split',self.root/'report.json')

    def test_split_fixture_has_all_labels_and_no_leakage(self):
        # Content is an opaque fixture token, not a real sentiment experiment.
        rows=[{'id':f'{group:02d}{label_index}', 'post_id':f'{group:03d}',
               'text':'Fixture '+hashlib.sha256(f'{group}:{label}'.encode()).hexdigest(),
               'label':label, 'annotation_status':'human_confirmed'}
              for group in range(9) for label_index,label in enumerate(['positive','neutral','negative'])]
        write_table(self.source,rows,list(rows[0]))
        report=self.root/'report.json'
        write_json(report,{'status':'confirmed','provenance':'real','labeled_sha256':file_hash(self.source),
                          'note':'Synthetic fixture; forged provenance only to check gate mechanics.'})
        result=split_confirmed(self.source,self.root/'split',report)
        frames,_,_=load_split_set(self.root/'split')
        self.assertEqual(sum(len(f) for f in frames.values()),len(rows))
        self.assertEqual(result['seed'],42)

    def test_near_duplicates_cannot_cross_splits(self):
        rows,fields=read_table('tests/fixtures/synthetic.csv')
        for row in rows:row['annotation_status']='human_confirmed'
        write_table(self.source,rows,fields+['annotation_status'])
        with self.assertRaisesRegex(ValueError,'gần trùng'):
            split_confirmed(self.source,self.root/'split',self.root/'unused_report.json')
        self.assertFalse((self.root/'split').exists())

    def test_quality_flags_do_not_decide_sentiment(self):
        self.assertIn('possible_tag_or_name_only_not_automatic_exclusion',flags('Nguyễn Văn A'))
        self.assertIn('excel_import_as_text_required',flags('=1+1'))
        self.assertEqual(flags('😢'),[])


if __name__=='__main__':unittest.main()
