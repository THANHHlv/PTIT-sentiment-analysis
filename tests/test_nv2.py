"""Focused checks for the NV2 preprocessing and feature contract."""

import unittest
import sys
from types import SimpleNamespace
from unittest.mock import patch
from sklearn.pipeline import Pipeline

from ptit_sentiment.preprocess import normalize_text
from ptit_sentiment.preprocess import TextPreprocessor
from ptit_sentiment.features import make_vectorizer


class NormalizationTests(unittest.TestCase):
    def test_whitespace_broken_character_and_url_punctuation(self):
        self.assertEqual(
            normalize_text("  Chưa\ufffd ổn\twww.example.org!  "),
            "chưa ổn urltoken !",
        )

    def test_slang_negation_and_emoji_are_preserved(self):
        self.assertEqual(normalize_text("SV ko thích 😢"), "sinh viên không thích 😢")

    def test_invalid_text_has_clear_error(self):
        with self.assertRaisesRegex(TypeError, "chuỗi Unicode"):
            normalize_text(None)


class TokenizationTests(unittest.TestCase):
    def test_same_transform_for_train_and_new_comment(self):
        fake = SimpleNamespace(word_tokenize=lambda text, format: text.replace("sinh viên", "sinh_viên"))
        with patch.dict(sys.modules, {"underthesea": fake}):
            processor = TextPreprocessor().fit(["SV ko thích"])
            self.assertEqual(processor.transform(["SV ko thích"]), ["sinh_viên không thích"])
            self.assertEqual(processor.transform(["SV ko thích"]), ["sinh_viên không thích"])

    def test_negation_cannot_be_stop_word(self):
        with self.assertRaisesRegex(ValueError, "phủ định"):
            TextPreprocessor(stop_words=["chưa"]).fit([])

    def test_stop_words_must_be_strings(self):
        with self.assertRaisesRegex(ValueError, "stop_words"):
            TextPreprocessor(stop_words=[None]).fit([])


class FeatureTests(unittest.TestCase):
    def test_bow_and_bigram(self):
        vectorizer = make_vectorizer("bow", (1, 2))
        matrix = vectorizer.fit_transform(["không tốt", "rất tốt"])
        self.assertIn("không tốt", vectorizer.vocabulary_)
        self.assertIn("rất tốt", vectorizer.vocabulary_)
        self.assertEqual(matrix.shape[0], 2)

    def test_tfidf_test_transform_keeps_vocabulary_and_idf(self):
        vectorizer = make_vectorizer("tfidf")
        vectorizer.fit(["không tốt", "rất tốt"])
        words = dict(vectorizer.vocabulary_)
        idf = vectorizer.idf_.copy()
        transformed = vectorizer.transform(["từ_mới tốt"])
        self.assertEqual(vectorizer.vocabulary_, words)
        self.assertEqual(vectorizer.idf_.tolist(), idf.tolist())
        self.assertNotIn("từ_mới", words)
        self.assertEqual(transformed.shape[1], len(words))

    def test_handoff_features_fit_only_train_and_reuse_for_new_comment(self):
        from features import build_feature_matrices

        fake = SimpleNamespace(word_tokenize=lambda text, format: text)
        with patch.dict(sys.modules, {"underthesea": fake}):
            bundle = build_feature_matrices(
                ["không tốt", "rất tốt"], ["chưa tốt"], ["từ_mới tốt"],
                kind="tfidf", ngram_range=(1, 2),
            )
            vocabulary = dict(bundle.vectorizer.vocabulary_)
            idf = bundle.vectorizer.idf_.copy()
            new_vector = bundle.transform_comment("rất tốt")
            self.assertEqual(bundle.X_train.shape[1], bundle.X_validation.shape[1])
            self.assertEqual(bundle.X_train.shape[1], bundle.X_test.shape[1])
            self.assertEqual(new_vector.shape[1], bundle.X_train.shape[1])
            self.assertNotIn("từ_mới", vocabulary)
            self.assertEqual(bundle.vectorizer.vocabulary_, vocabulary)
            self.assertEqual(bundle.vectorizer.idf_.tolist(), idf.tolist())
    def test_handoff_features_allow_train_only(self):
        from features import build_feature_matrices

        fake = SimpleNamespace(word_tokenize=lambda text, format: text)
        with patch.dict(sys.modules, {"underthesea": fake}):
            bundle = build_feature_matrices(["không tốt", "rất tốt"], [], [])
            self.assertEqual(bundle.X_validation.shape, (0, bundle.X_train.shape[1]))
            self.assertEqual(bundle.X_test.shape, (0, bundle.X_train.shape[1]))
            self.assertEqual(bundle.transform_comment("rất tốt").shape[1], bundle.X_train.shape[1])
    def test_invalid_ngram_range(self):
        with self.assertRaisesRegex(ValueError, "ngram_range"):
            make_vectorizer("bow", (2, 2))

    def test_pipeline_reuses_train_features_for_new_text(self):
        fake = SimpleNamespace(word_tokenize=lambda text, format: text)
        with patch.dict(sys.modules, {"underthesea": fake}):
            pipeline = Pipeline([
                ("preprocess", TextPreprocessor()),
                ("vectorizer", make_vectorizer("tfidf", (1, 2))),
            ])
            train_matrix = pipeline.fit_transform(["ko tốt", "rất tốt"])
            vectorizer = pipeline.named_steps["vectorizer"]
            vocabulary = dict(vectorizer.vocabulary_)
            idf = vectorizer.idf_.copy()
            new_matrix = pipeline.transform(["chưa tốt từ_mới"])
            self.assertEqual(train_matrix.shape[1], new_matrix.shape[1])
            self.assertEqual(vectorizer.vocabulary_, vocabulary)
            self.assertEqual(vectorizer.idf_.tolist(), idf.tolist())
            self.assertNotIn("từ_mới", vocabulary)


if __name__ == "__main__":
    unittest.main()
