"""Check a saved word/word-char pipeline's train-only vocabulary and IDF."""
import argparse

import joblib
import numpy as np
from scipy.sparse import issparse
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion

from ptit_sentiment.common import file_hash, read_json, run_cli, write_json
from ptit_sentiment.data.split import load_split_set
from ptit_sentiment.labels import LABELS


def audit(model_path, splits_dir, output):
    """Inspect fitted parameters without refitting or predicting test labels."""
    model = joblib.load(model_path)
    metadata = read_json(model_path.with_suffix(".metadata.json"))
    frames, _, digest = load_split_set(splits_dir)
    if metadata["split_manifest_sha256"] != digest:
        raise ValueError("Model and data manifest hashes differ.")
    processor, vectorizer = model.named_steps["preprocess"], model.named_steps["vectorizer"]
    train_tokens = processor.transform(frames["train"]["text"].tolist())
    validation_tokens = processor.transform(frames["validation"]["text"].tolist())
    branches = vectorizer.transformer_list if isinstance(vectorizer, FeatureUnion) else [("word", vectorizer)]
    checks = {}
    for name, branch in branches:
        vocabulary_before = dict(branch.vocabulary_)
        idf_before = branch.idf_.copy() if hasattr(branch, "idf_") else None
        analyzer = branch.build_analyzer()
        train_terms = set(term for text in train_tokens for term in analyzer(text))
        assert set(vocabulary_before) <= train_terms, f"{name}: vocabulary contains terms absent from train"
        train_matrix = branch.transform(train_tokens)
        heldout_matrix = branch.transform(validation_tokens)
        assert issparse(train_matrix) and issparse(heldout_matrix)
        assert not train_matrix.data.size or train_matrix.data.min() >= 0
        if idf_before is not None:
            document_frequency = np.zeros(len(vocabulary_before), dtype=int)
            for text in train_tokens:
                for term in set(analyzer(text)):
                    if term in vocabulary_before:
                        document_frequency[vocabulary_before[term]] += 1
            expected = (np.log((1 + len(train_tokens)) / (1 + document_frequency)) + 1
                        if branch.smooth_idf else np.log(len(train_tokens) / document_frequency) + 1)
            np.testing.assert_allclose(idf_before, expected)
            np.testing.assert_array_equal(branch.idf_, idf_before)
        assert branch.vocabulary_ == vocabulary_before
        checks[name] = {"shape_train": list(train_matrix.shape), "nnz_train": train_matrix.nnz,
                        "sparse_nonnegative": True, "vocabulary_subset_of_train": True,
                        "idf_matches_train_only": idf_before is not None,
                        "heldout_did_not_change_parameters": True}
    train_matrix = vectorizer.transform(train_tokens)
    validation_matrix = vectorizer.transform(validation_tokens)
    assert issparse(train_matrix) and issparse(validation_matrix)
    raw_predictions = model.predict(frames["validation"]["text"].tolist())
    score = float(f1_score(frames["validation"]["label"], raw_predictions,
                          labels=list(LABELS), average="macro", zero_division=0))
    assert np.isclose(score, metadata["validation_macro_f1"])
    write_json(output, {"model_name": metadata["model_name"], "artifact_sha256": file_hash(model_path),
                        "split_manifest_sha256": digest, "branches": checks,
                        "train_shape": list(train_matrix.shape), "validation_shape": list(validation_matrix.shape),
                        "zero_train_vectors": int((train_matrix.getnnz(axis=1) == 0).sum()),
                        "zero_validation_vectors": int((validation_matrix.getnnz(axis=1) == 0).sum()),
                        "saved_raw_pipeline_validation_macro_f1": score,
                        "matches_selection_score": True, "test_predictions_computed": False})
    print(f"Train-only feature audit passed: {metadata['model_name']}")


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit(args.model, args.splits_dir, args.output)


if __name__ == "__main__":
    run_cli(main)
