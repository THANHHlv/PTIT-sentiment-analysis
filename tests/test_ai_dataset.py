"""Fixture tạm kiểm tra nguồn AI, đối soát và nhóm chia; không train."""
import hashlib
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ptit_sentiment.common import file_hash, read_json, write_json
from ptit_sentiment.data.ai_dataset import checked_origins, connected_posts, export_labels, split_ai_dataset
from ptit_sentiment.data.annotation_review import audit, read_table, record, state, write_table
from ptit_sentiment.data.split import embedded_metadata_hash, load_split_set


class AiDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.work = self.root / 'work'
        self.raw = self.root / 'raw.csv'
        self.rows = [{'id': f'{g:03d}{j}', 'post_id': f'{g:03d}',
                      'text': 'Fixture ' + hashlib.sha256(f'{g}:{j}'.encode()).hexdigest()}
                     for g in range(9) for j in range(3)]
        self.rows += [{'id': 'unknown', 'post_id': '010', 'text': 'Ambiguous fixture'},
                      {'id': 'advert', 'post_id': '011', 'text': 'Excluded fixture'}]
        write_table(self.raw, self.rows, ['id', 'post_id', 'text'])
        audit(self.raw, self.work)

    def finish(self):
        base, _, manifest = state(self.work)
        decisions = [{'seq': i, 'suggested_label': ('positive', 'neutral', 'negative')[(i-1) % 3],
                      'review_priority': 'low', 'reason': 'Explicit fixture decision', 'action': 'keep'}
                     for i, _ in enumerate(base, 1)]
        decisions[-2].update(suggested_label='', review_priority='high', reason='Ambiguous fixture')
        decisions[-1].update(suggested_label='', review_priority='high', action='exclude')
        p = self.root / 'batch.json'
        write_json(p, {'base_sha256': manifest['base_sha256'], 'decisions': decisions})
        record(self.work, p)

    def test_incomplete_checkpoint_is_not_finalized(self):
        with self.assertRaisesRegex(ValueError, 'Chưa đọc hết'):
            export_labels(self.work)
        self.assertFalse((self.work / 'labeled_comments.csv').exists())

    def test_full_partition_and_origins_roundtrip(self):
        before = file_hash(self.raw)
        self.finish()
        report = split_ai_dataset(self.work, self.root / 'splits')
        frames, metadata, _ = load_split_set(self.root / 'splits')
        labeled, _ = read_table(self.work / 'labeled_comments.csv')
        unresolved, _ = read_table(self.work / 'unresolved.csv')
        excluded, _ = read_table(self.work / 'excluded_comments.csv')
        self.assertEqual((len(labeled), len(unresolved), len(excluded)), (27, 1, 1))
        self.assertEqual(sum(len(f) for f in frames.values()), 27)
        self.assertEqual(before, file_hash(self.raw))
        self.assertFalse(metadata['human_labels_confirmed'])
        self.assertEqual({r['annotation_status'] for r in labeled}, {'ai_labeled'})
        self.assertEqual({r['label_source'] for r in labeled}, {'ai'})
        self.assertEqual(unresolved[0]['label'], '')
        self.assertEqual(report['checks']['post_intersections'], 0)
        with self.assertRaisesRegex(ValueError, 'đích đã tồn tại'):
            split_ai_dataset(self.work, self.root / 'splits')

    def test_pending_and_fake_human_rejected(self):
        row = {'label_source': 'ai', 'annotation_status': 'ai_labeled'}
        checked_origins(pd.DataFrame([row]))
        for status in ['pending_review', 'unresolved', 'human_confirmed']:
            with self.assertRaisesRegex(ValueError, 'Nguồn/trạng thái'):
                checked_origins(pd.DataFrame([{**row, 'annotation_status': status}]))

    def test_compact_metadata_roundtrip_and_tampering(self):
        self.finish()
        directory = self.root / 'splits'
        split_ai_dataset(self.work, directory)
        manifest = read_json(directory / 'manifest.json')
        groups, _ = read_table(directory / 'split_groups.csv')
        pairs, _ = read_table(directory / 'near_duplicate_constraints.csv')
        embedded = {'labeling_report': read_json(directory / 'labeling_report.json'),
                    'split_groups': groups, 'near_duplicate_constraints': pairs}
        manifest.update(storage_layout='compact_v1', embedded_metadata=embedded,
                        embedded_metadata_sha256=embedded_metadata_hash(embedded))
        write_json(directory / 'manifest.json', manifest)
        for name in ['labeling_report.json', 'split_groups.csv', 'near_duplicate_constraints.csv', 'metadata.json']:
            (directory / name).unlink()
        frames, meta, _ = load_split_set(directory)
        self.assertEqual(sum(len(f) for f in frames.values()), 27)
        self.assertFalse(meta['human_labels_confirmed'])
        manifest['embedded_metadata']['labeling_report']['human_labels_confirmed'] = True
        write_json(directory / 'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'nhúng'):
            load_split_set(directory)

    def test_transitive_near_and_historical_post_links(self):
        rows = [{'id': str(i), 'post_id': f'{i:03d}', 'text': f'Fixture {i}'} for i in range(4)]
        pairs = [{'id_a': '0', 'id_b': '1'}, {'id_a': '1', 'id_b': '2'}]
        history = [{'canonical_id': '2', 'post_id': '003'}]
        groups = connected_posts(rows, rows, pairs, history)
        self.assertEqual(len(set(groups.values())), 1)
        self.assertEqual(rows[0]['post_id'], '000')
        with self.assertRaisesRegex(ValueError, 'Thiếu post_id'):
            connected_posts([{**rows[0], 'post_id': ''}], rows, pairs, history)

    def test_human_input_kept_despite_different_ai_proposal(self):
        # Temporary fixture declaration checks preservation, not real confirmation.
        temp = self.root / 'human'
        row = {**self.rows[0], 'label': 'positive', 'annotation_status': 'human_confirmed',
               'label_source': 'fixture_human', 'reviewer': 'fixture_reviewer'}
        write_table(self.raw, [row], list(row))
        audit(self.raw, temp)
        _, _, manifest = state(temp)
        p = self.root / 'human_batch.json'
        write_json(p, {'base_sha256': manifest['base_sha256'], 'decisions': [
            {'seq': 1, 'suggested_label': 'negative', 'review_priority': 'high', 'reason': 'Fixture disagreement'}]})
        record(temp, p)
        labels, report = export_labels(temp)
        self.assertEqual((labels[0]['label'], labels[0]['label_source'], labels[0]['annotation_status']),
                         ('positive', 'fixture_human', 'human_confirmed'))
        self.assertEqual(report['human_confirmed_rows'], 1)


if __name__ == '__main__':
    unittest.main()
