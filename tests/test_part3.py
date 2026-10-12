"""Meaningful integration guards for classical selection and convergence."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import warnings

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.exceptions import ConvergenceWarning

from ptit_sentiment.common import read_json
from ptit_sentiment.data.split import save_splits
from ptit_sentiment.models.classical import MODEL_NAMES
from ptit_sentiment.predict import load_artifact
from ptit_sentiment.preprocessing.tokenize import TextPreprocessor
from ptit_sentiment.training.train_classical import fit_classifier, train_models


class Part3Test(unittest.TestCase):
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
                    self.assertEqual(len(search["trials"]), 6)
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
