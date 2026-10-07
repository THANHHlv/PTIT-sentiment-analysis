"""Fixture tạm chỉ kiểm tra importer/gán nhãn; không phải dữ liệu Facebook."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from ptit_sentiment.common import read_json, write_json
from ptit_sentiment.data.collect import collect
from ptit_sentiment.data.labeling import prepare, merge
from ptit_sentiment.data.split import save_splits, load_split_set


class RealWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "page.csv"
        pd.DataFrame([
            {"id": "001", "post_id": "010", "text": "Fixture A"},
            {"id": "002", "post_id": "011", "text": "Fixture B"},
            {"id": "003", "post_id": "012", "text": "Fixture C"},
        ]).to_csv(self.raw, index=False)
        self.config = self.root / "collection.yaml"
        self.source = {"source_id": "test_fixture", "source_url": "https://example.invalid/fixture",
                       "permission_note": "Temporary test fixture, not PTIT data",
                       "exports": [str(self.raw)]}

    def collect_fixture(self, target=2000):
        self.config.write_text(yaml.safe_dump({"sources": [self.source], "target_comments": target,
                                               "batch_size": 1}), encoding="utf-8")
        return collect(self.config, self.root / "collected")

    def annotation_fixture(self):
        assignment = self.root / "assignment"
        prepare(self.raw, assignment, cross_fraction=1.0, provenance="synthetic")
        return assignment

    def confirm(self, assignment):
        for person in range(1, 5):
            path = assignment / f"annotator_{person}.csv"
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            frame["label"] = frame["id"].map({"001": "positive", "002": "neutral", "003": "negative"})
            frame["confirmed"] = "yes"
            frame.to_csv(path, index=False)

    def test_resume_caps_and_preserves_string_ids(self):
        first = self.collect_fixture(target=2)
        self.assertEqual(first["valid_unique_ids"], 2)
        resumed = self.collect_fixture(target=3)
        self.assertEqual(resumed["valid_unique_ids"], 3)
        final = self.collect_fixture(target=3)
        self.assertEqual(final["sources"][0]["imported_this_run"], 0)
        data = pd.read_csv(self.root / "collected/comments.csv", dtype=str)
        self.assertEqual(data["id"].tolist(), ["001", "002", "003"])

    def test_changed_checkpoint_rejected_without_loss(self):
        self.collect_fixture()
        with self.raw.open("a", encoding="utf-8") as stream:
            stream.write("004,013,Fixture D\n")
        with self.assertRaises(ValueError):
            self.collect_fixture()
        report = read_json(self.root / "collected/collection_report.json")
        self.assertEqual(report["valid_unique_ids"], 3)
        self.assertIn("checkpoint", report["sources"][0]["error"])

    def test_duplicates_conflict_and_missing_columns(self):
        self.collect_fixture()
        second = self.root / "page2.csv"
        pd.DataFrame([{"id": "001", "post_id": "010", "text": "Fixture A"},
                      {"id": "002", "post_id": "011", "text": "Changed"}]).to_csv(second, index=False)
        self.source["exports"].append(str(second))
        report = self.collect_fixture()
        self.assertEqual(report["valid_unique_ids"], 3)
        self.assertEqual(report["duplicate_id_rows"], 1)
        self.assertEqual(len(report["rejected_rows"]), 1)
        bad = self.root / "bad.csv"
        bad.write_text("body\nmissing identifiers\n", encoding="utf-8")
        self.source["exports"].append(str(bad))
        with self.assertRaises(ValueError):
            self.collect_fixture()
        self.assertIn("thiếu cột", read_json(self.root / "collected/collection_report.json")["sources"][0]["error"])

    def test_suggestions_never_become_gold(self):
        assignment = self.annotation_fixture()
        for person in range(1, 5):
            path = assignment / f"annotator_{person}.csv"
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            frame["suggested_label"] = "positive"
            frame.to_csv(path, index=False)
        output = self.root / "unconfirmed"
        report = merge(assignment, output)
        self.assertEqual(report["status"], "needs_human_review")
        self.assertFalse((output / "labeled.csv").exists())

    def test_agreement_confirmed_and_text_immutable(self):
        assignment = self.annotation_fixture()
        self.confirm(assignment)
        report = merge(assignment, self.root / "confirmed")
        self.assertEqual(report["status"], "confirmed")
        self.assertEqual(report["label_counts"], {"positive": 1, "neutral": 1, "negative": 1})
        path = assignment / "annotator_1.csv"
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        frame.loc[0, "text"] = "Tampered"
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "đã sửa"):
            merge(assignment, self.root / "tampered")

    def test_disagreement_resolution_template(self):
        assignment = self.annotation_fixture()
        self.confirm(assignment)
        path = assignment / "annotator_1.csv"
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        frame.loc[0, "label"] = "negative" if frame.loc[0, "label"] != "negative" else "positive"
        identifier = frame.loc[0, "id"]
        frame.to_csv(path, index=False)
        output = self.root / "review"
        report = merge(assignment, output)
        self.assertEqual(report["status"], "needs_human_review")
        resolutions = pd.read_csv(output / "needs_review.csv", dtype=str, keep_default_na=False)
        self.assertEqual(resolutions["id"].tolist(), [identifier])
        resolutions["final_label"] = "neutral"
        resolutions["confirmed"] = "yes"
        resolutions["reviewer"] = "test reviewer"
        resolutions["reason"] = "Test-only arbitration"
        resolution_path = self.root / "resolutions.csv"
        resolutions.to_csv(resolution_path, index=False)
        # Threshold lowered only to test arbitration; production default remains .8.
        resolved = merge(assignment, self.root / "resolved", resolution_path, min_cross_agreement=0.5)
        self.assertEqual(resolved["status"], "confirmed")
        self.assertEqual(resolved["unresolved_samples"], 0)

    def test_real_split_requires_matching_confirmation(self):
        with self.assertRaisesRegex(ValueError, "labeling-report"):
            save_splits(self.raw, self.root / "split", (.7, .15, .15), 42, 20, "real")
        input_path = Path("tests/fixtures/synthetic.csv")
        audit = self.root / "audit.json"
        write_json(audit, {"status": "confirmed", "provenance": "real", "labeled_sha256": "wrong"})
        with self.assertRaisesRegex(ValueError, "phiên bản"):
            save_splits(input_path, self.root / "split", (.7, .15, .15), 42, 20, "real", audit)
        from ptit_sentiment.common import file_hash
        # Forged provenance in a temporary test checks gate mechanics, not source authenticity.
        write_json(audit, {"status": "confirmed", "provenance": "real", "labeled_sha256": file_hash(input_path)})
        save_splits(input_path, self.root / "split", (.7, .15, .15), 42, 20, "real", audit)
        load_split_set(self.root / "split")
        write_json(self.root / "split/labeling_report.json", {"status": "changed"})
        with self.assertRaisesRegex(ValueError, "thay đổi"):
            load_split_set(self.root / "split")


if __name__ == "__main__":
    unittest.main()