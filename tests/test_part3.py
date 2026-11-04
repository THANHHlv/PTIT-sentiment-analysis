"""Meaningful integration guards for classical selection and convergence."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import warnings

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score

from ptit_sentiment.common import read_json
from ptit_sentiment.data.split import save_splits
from ptit_sentiment.models.classical import MODEL_NAMES
from ptit_sentiment.predict import load_artifact
from ptit_sentiment.preprocessing.tokenize import TextPreprocessor
from ptit_sentiment.training.train_classical import fit_classifier, train_models


class Part3Test(unittest.TestCase):
    def test_shared_tfidf_options_preserved_without_breaking_bow(self):
        """A TF-IDF option in YAML must survive selection and not reach CountVectorizer."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_splits("tests/fixtures/synthetic.csv", root / "splits", (.7, .15, .15), 42, 500, "synthetic")
            config = yaml.safe_load(Path("configs/part3.yaml").read_text(encoding="utf-8"))
            config["vectorizer"].update(sublinear_tf=True, norm="l2", use_idf=True, smooth_idf=True)
            config["grid"].pop("sublinear_tf")
            config["grid"].update(ngram_range=[[1, 1]], nb_alpha=[0.1], svm_c=[1.0])
            (root / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
            fake = SimpleNamespace(word_tokenize=lambda text, format: text)
            with patch.dict("sys.modules", {"underthesea": fake}):
                train_models(root / "splits", root / "config.yaml", root / "models", root / "logs")
                for name in MODEL_NAMES:
                    model, metadata = load_artifact(root / "models" / (name + ".joblib"))
                    if name.startswith("tfidf"):
                        self.assertTrue(model.named_steps["vectorizer"].sublinear_tf)
                        self.assertTrue(metadata["selected_parameters"]["sublinear_tf"])
                    self.assertAlmostEqual(metadata["validation_macro_f1"], metadata["validation_metrics"]["macro_f1"])
                _, baseline_metadata = load_artifact(root / "models/majority_baseline.joblib")
                self.assertEqual(set(baseline_metadata["validation_metrics"]["per_label"]),
                                 {"positive", "neutral", "negative"})

    def test_emoji_grid_cache_matches_saved_raw_text_pipeline(self):
        """Emoji candidates must use separate tokens and retain their selected mode."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = pd.read_csv("tests/fixtures/synthetic.csv", dtype=str, keep_default_na=False)
            source["text"] = source["text"] + " 😢"
            source.to_csv(root / "source.csv", index=False, encoding="utf-8")
            save_splits(root / "source.csv", root / "splits", (.7, .15, .15), 42, 500, "synthetic")
            config = yaml.safe_load(Path("configs/part3.yaml").read_text(encoding="utf-8"))
            config.update(models=["tfidf_svm"], model_aliases={"tfidf_svm": "tfidf_svm_extended"})
            config["grid"].update(ngram_range=[[1, 1]], svm_c=[1.0],
                                  emoji=["keep", "keep_and_map"],
                                  svm_class_weight=[None, "balanced"], tfidf_mode=["word", "word_char"])
            (root / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
            fake = SimpleNamespace(word_tokenize=lambda text, format: text)
            with patch.dict("sys.modules", {"underthesea": fake}):
                train_models(root / "splits", root / "config.yaml", root / "models", root / "logs")
                search = read_json(root / "logs/tfidf_svm_extended_validation_search.json")
                self.assertEqual(len(search["trials"]), 8)
                word_trials = [t for t in search["trials"] if t["parameters"]["tfidf_mode"] == "word"
                               and t["parameters"]["class_weight"] is None]
                dimensions = {t["parameters"]["emoji"]: t["train_shape"][1] for t in word_trials}
                self.assertGreater(dimensions["keep_and_map"], dimensions["keep"])
                model, metadata = load_artifact(root / "models/tfidf_svm_extended.joblib")
                validation = pd.read_csv(root / "splits/validation.csv", dtype=str)
                predicted = model.predict(validation["text"])
                actual_score = f1_score(validation["label"], predicted,
                                        labels=["positive", "neutral", "negative"], average="macro", zero_division=0)
                self.assertAlmostEqual(actual_score, metadata["validation_macro_f1"])
                self.assertEqual(model.named_steps["preprocess"].emoji, metadata["selected_parameters"]["emoji"])
                self.assertEqual(model.named_steps["classifier"].class_weight,
                                 metadata["selected_parameters"]["class_weight"])

    def test_invalid_or_removed_comment_rejected_before_segmentation(self):
        for value in (None, "", " ", "\ufffd\u200b"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                TextPreprocessor().fit_transform([value])

    def test_convergence_retry_and_fail_closed(self):
        class Classifier:
            max_iter = 1
            n_iter_ = 1
            def set_params(self, **kwargs):
                self.max_iter = kwargs["max_iter"]
            def fit(self, X, y):
                if self.max_iter == 1:
                    warnings.warn("not converged", ConvergenceWarning)
        attempts = fit_classifier(Classifier(), csr_matrix([[1]]), ["neutral"], 2)
        self.assertEqual(len(attempts), 2)
        self.assertFalse(attempts[-1]["convergence_warnings"])
        with self.assertRaisesRegex(ValueError, "hội tụ"):
            fit_classifier(Classifier(), csr_matrix([[1]]), ["neutral"], 1)

    def test_all_models_train_only_and_reload_with_selection_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_splits("tests/fixtures/synthetic.csv", root / "splits", (.7, .15, .15), 42, 500, "synthetic")
            fake = SimpleNamespace(word_tokenize=lambda text, format: text)
            with patch.dict("sys.modules", {"underthesea": fake}):
                train_models(root / "splits", "configs/part3.yaml", root / "models", root / "logs")
                lock = read_json(root / "logs/selection_lock.json")
                self.assertFalse(lock["test_predictions_computed"])
                for name in MODEL_NAMES:
                    model, metadata = load_artifact(root / "models" / (name + ".joblib"))
                    self.assertFalse(metadata["refit_train_validation"])
                    search = read_json(root / "logs" / (name + "_validation_search.json"))
                    self.assertEqual(len(search["trials"]), 4)
                    self.assertEqual(metadata["validation_macro_f1"], max(t["validation_macro_f1"] for t in search["trials"]))
                    before = dict(model.named_steps["vectorizer"].vocabulary_)
                    idf = getattr(model.named_steps["vectorizer"], "idf_", np.array([])).copy()
                    prediction = model.predict(["chưa thích unseenpart3token"])
                    self.assertEqual(before, model.named_steps["vectorizer"].vocabulary_)
                    np.testing.assert_array_equal(idf, getattr(model.named_steps["vectorizer"], "idf_", np.array([])))
                    np.testing.assert_array_equal(prediction, joblib.load(root / "models" / (name + ".joblib")).predict(["chưa thích unseenpart3token"]))
                path = root / "models/bow_nb.joblib"
                path.write_bytes(path.read_bytes() + b"tampered")
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    load_artifact(path)


if __name__ == "__main__":
    unittest.main()
