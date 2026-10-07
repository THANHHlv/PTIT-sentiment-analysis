"""Kiểm tra ràng buộc chống rò rỉ và dữ liệu sai bằng unittest."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ptit_sentiment.common import write_json
from ptit_sentiment.data.split import grouped_split, load_split_set, save_splits
from ptit_sentiment.data.validate import read_csv, validate_frame
from ptit_sentiment.evaluation.compare import compare_results
from ptit_sentiment.evaluation.metrics import compute_metrics
from ptit_sentiment.features.vectorizers import make_vectorizer
from ptit_sentiment.labels import LABELS
from ptit_sentiment.predict import load_artifact
from ptit_sentiment.preprocessing.normalize import normalize_text
from ptit_sentiment.preprocessing.tokenize import TextPreprocessor


class ContractsTest(unittest.TestCase):
    def setUp(self):
        self.frame = read_csv("tests/fixtures/synthetic.csv")

    def test_reproducible_groups_and_disjoint_text(self):
        first, second = grouped_split(self.frame), grouped_split(self.frame)
        for name in first:
            self.assertEqual(first[name]["id"].tolist(), second[name]["id"].tolist())
            self.assertEqual(set(first[name]["label"]), set(LABELS))
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            self.assertFalse(set(first[left]["post_id"]) & set(first[right]["post_id"]))
            self.assertFalse(set(first[left]["text"]) & set(first[right]["text"]))

    def test_too_small_does_not_fallback(self):
        with self.assertRaisesRegex(ValueError, "quá nhỏ"):
            grouped_split(self.frame.iloc[:3])

    def test_duplicates_case_spacing(self):
        duplicate = self.frame.iloc[[0]].copy()
        duplicate["id"] = "extra"
        duplicate["text"] = "  " + duplicate["text"].iloc[0].upper() + "  "
        with self.assertRaisesRegex(ValueError, "Trùng bình luận"):
            validate_frame(pd.concat([self.frame, duplicate], ignore_index=True))

    def test_empty_invalid_label_duplicate_id(self):
        for column, value, message in (("text", "  ", "rỗng"), ("label", "unknown", "Nhãn"),
                                        ("id", self.frame["id"].iloc[1], "Trùng id")):
            frame = self.frame.copy()
            frame.loc[0, column] = value
            with self.assertRaisesRegex(ValueError, message):
                validate_frame(frame)

    def test_manifest_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            save_splits("tests/fixtures/synthetic.csv", directory, [0.7, 0.15, 0.15], 42, 100, "synthetic")
            path = Path(directory) / "test.csv"
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "đã đổi"):
                load_split_set(directory)

    def test_negation_stopwords(self):
        self.assertIn("không", normalize_text("SV ko thích"))
        with self.assertRaisesRegex(ValueError, "phủ định"):
            TextPreprocessor(stop_words=["không tốt"]).fit(["không tốt"])

    def test_unseen_token_does_not_change_vocabulary(self):
        for kind in ("bow", "tfidf"):
            vectorizer = make_vectorizer(kind)
            vectorizer.fit(["train_token common", "another common"])
            vocabulary = dict(vectorizer.vocabulary_)
            transformed = vectorizer.transform(["validation_only common"])
            self.assertEqual(vectorizer.vocabulary_, vocabulary)
            self.assertNotIn("validation_only", vectorizer.vocabulary_)
            self.assertEqual(transformed.shape[1], len(vocabulary))

    def test_macro_fixed_labels(self):
        result = compute_metrics(list(LABELS), ["positive"] * 3)
        self.assertEqual(result["per_label"]["negative"]["f1"], 0.0)
        self.assertAlmostEqual(result["macro_f1"], 1 / 6)
        self.assertEqual(result["confusion_matrix"]["rows"], "true_label")

    def test_missing_artifact_clear_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "Artifact chưa tồn tại"):
            load_artifact("does-not-exist.joblib")

    def test_compare_mismatched_split(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index in range(2):
                path = Path(directory) / f"metrics{index}.json"
                write_json(path, {"model_name": f"model{index}", "split_manifest_sha256": str(index),
                                 "evaluation_data_sha256": "a", "split": "test", "provenance": "synthetic",
                                 "n_samples": 3, "accuracy": 0, "macro_precision": 0, "macro_recall": 0, "macro_f1": 0})
                paths.append(path)
            with self.assertRaisesRegex(ValueError, "khác nhau"):
                compare_results(paths, Path(directory) / "comparison.csv")



    def test_phobert_path_preserves_case_slang_and_restores_cwd(self):
        # Mock segmenter: kiểm tra đường xử lý, không khởi tạo JVM hay tải model.
        import os
        import sys
        import types
        from unittest.mock import patch
        from ptit_sentiment.preprocessing.tokenize import PhoBERTPreprocessor

        class FakeSegmenter:
            def __init__(self, annotators, save_dir):
                self.asserted_annotators = annotators
                os.chdir(save_dir)

            def word_segment(self, text):
                return [text]

        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("VnCoreNLP-1.2.jar", "models/wordsegmenter/vi-vocab",
                         "models/wordsegmenter/wordsegmenter.rdr"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            fake = types.SimpleNamespace(VnCoreNLP=FakeSegmenter)
            with patch.dict(sys.modules, {"py_vncorenlp": fake}):
                processor = PhoBERTPreprocessor(root)
                self.assertEqual(Path.cwd(), previous)
                self.assertEqual(processor.transform(["  SV ko thích 😢 https://example.org  "]),
                                 ["SV ko thích 😢 https://example.org"])

    def test_phobert_invalid_length_before_heavy_imports(self):
        from ptit_sentiment.training.train_phobert import train_phobert
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "invalid.yaml"
            config.write_text("max_length: 512", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "max_length"):
                train_phobert("unused", config, Path(directory) / "unused")

    def test_evaluate_rejects_other_training_manifest(self):
        import joblib
        from ptit_sentiment.evaluation.evaluate import evaluate_model
        from ptit_sentiment.labels import LABEL_TO_ID
        from ptit_sentiment.models.classical import fit_baseline
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            splits = root / "splits"
            save_splits("tests/fixtures/synthetic.csv", splits, [0.7, 0.15, 0.15], 42, 100, "synthetic")
            model = root / "baseline.joblib"
            joblib.dump(fit_baseline(self.frame["text"].tolist(), self.frame["label"].tolist()), model)
            write_json(model.with_suffix(".metadata.json"), {
                "label_to_id": LABEL_TO_ID, "split_manifest_sha256": "other",
                "kind": "classical", "model_name": "baseline", "baseline": True})
            with self.assertRaisesRegex(ValueError, "khác bộ chia"):
                evaluate_model(model, splits, root / "evaluation")


    def test_sensitive_controls_and_configurable_slang(self):
        raw = "SV iu bài học\x00 mail a@example.com gọi 0912345678 @test"
        cleaned = normalize_text(raw, slang_map={"iu": "yêu"})
        self.assertIn("yêu", cleaned)
        self.assertIn("emailtoken", cleaned)
        self.assertIn("phonetoken", cleaned)
        self.assertIn("usertoken", cleaned)
        self.assertNotIn("a@example.com", cleaned)
        self.assertNotIn("\x00", cleaned)
        self.assertIn("SV", raw)

    def test_truncation_boundary_and_counts(self):
        from ptit_sentiment.preprocessing.phobert_input import encode_with_statistics
        class Tokenizer:
            truncation_side = "right"
            def __call__(self, texts, truncation=False, max_length=None, **kwargs):
                rows = [[0] + list(range(len(text.split()))) + [2] for text in texts]
                return {"input_ids": [row[:max_length] if truncation else row for row in rows]}
        encoding, stats = encode_with_statistics(Tokenizer(), ["a b", "a b c"], 4)
        self.assertEqual(stats["truncated_count"], 1)
        self.assertEqual(stats["truncated_samples"], [{"index": 1, "original_tokens": 5}])
        self.assertEqual([len(row) for row in encoding["input_ids"]], [4, 4])


if __name__ == "__main__":
    unittest.main()
