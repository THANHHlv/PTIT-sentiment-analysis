"""Audit the actual NV2 handoff and fixed real splits; never train a classifier."""
import argparse
from pathlib import Path
import sys
from time import perf_counter
import zipfile

import numpy as np
from scipy.sparse import issparse

# features.py is the existing NV2 handoff outside the installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features import build_feature_matrices
from ptit_sentiment.common import file_hash, load_config, run_cli, write_json
from ptit_sentiment.data.split import load_split_set
from ptit_sentiment.data.validate import read_csv, statistics
from ptit_sentiment.preprocess import TextPreprocessor, preprocess_text


def review(splits_dir, source, config_path, output):
    """Verify provenance, sparse features, row alignment, train-only vocabulary/IDF."""
    if Path(output).exists():
        raise ValueError("Audit output already exists; choose a new path.")
    clock = perf_counter()
    paths = [Path(source), *Path(splits_dir).glob("*")]
    before = {str(p): file_hash(p) for p in paths if p.is_file()}
    frames, manifest, digest = load_split_set(splits_dir)
    original = read_csv(source)
    assert file_hash(source) == manifest["source_sha256"]
    import pandas as pd
    joined = pd.concat(list(frames.values()), ignore_index=True)
    assert original.sort_values("id").reset_index(drop=True).equals(
        joined[original.columns].sort_values("id").reset_index(drop=True))
    config = load_config(config_path)
    checks = []
    for kind in ("bow", "tfidf"):
        for ngram in ((1, 1), (1, 2)):
            bundle = build_feature_matrices(
                *(frames[name]["text"] for name in ("train", "validation", "test")),
                kind=kind, ngram_range=ngram,
                preprocessing_options=config["preprocessing"],
                vectorizer_options=config["vectorizer"],
            )
            vectorizer = bundle.vectorizer
            vocabulary = dict(vectorizer.vocabulary_)
            idf = vectorizer.idf_.copy() if kind == "tfidf" else None
            matrices = {name: getattr(bundle, "X_" + name) for name in frames}
            for name, matrix in matrices.items():
                assert issparse(matrix) and matrix.shape[0] == len(frames[name])
                assert not matrix.nnz or matrix.data.min() >= 0
                # Each selected row must match transformation of the same raw id/text.
                for index in (0, len(frames[name]) // 2, len(frames[name]) - 1):
                    single = bundle.transform_comment(frames[name].iloc[index]["text"])
                    assert (single != matrix[index]).nnz == 0
            train_processed = bundle.preprocessor.transform(frames["train"]["text"])
            analyzer = vectorizer.build_analyzer()
            train_terms = {term for text in train_processed for term in analyzer(text)}
            assert set(vocabulary) <= train_terms
            if idf is not None:
                from ptit_sentiment.features import make_vectorizer
                counts = make_vectorizer("bow", ngram, vocabulary=vocabulary).transform(train_processed)
                document_frequency = np.asarray((counts > 0).sum(axis=0)).ravel()
                np.testing.assert_allclose(idf, np.log((1 + len(train_processed)) / (1 + document_frequency)) + 1)
            sentinel = "nv3unseenvalidationtoken987654"
            assert sentinel not in train_terms
            vectorizer.transform([sentinel])
            for name in ("validation", "test"):
                vectorizer.transform(bundle.preprocessor.transform(frames[name]["text"]))
            assert vectorizer.vocabulary_ == vocabulary
            if idf is not None:
                np.testing.assert_array_equal(idf, vectorizer.idf_)
            checks.append({"kind": kind, "ngram_range": list(ngram),
                           "shapes": {n: list(x.shape) for n, x in matrices.items()},
                           "nnz": {n: x.nnz for n, x in matrices.items()},
                           "zero_vectors": {n: int((x.getnnz(axis=1) == 0).sum()) for n, x in matrices.items()},
                           "sparse_nonnegative": True, "row_alignment": True,
                           "vocabulary_train_only": True, "idf_train_only": kind == "tfidf",
                           "vocabulary_idf_unchanged_after_heldout": True})
    invalid = []
    for value in (None, "", "   ", "\ufffd\u200b\ufeff", "😢"):
        options = {"emoji": "remove"} if value == "😢" else {}
        try:
            TextPreprocessor(**options).fit_transform([value])
        except ValueError:
            invalid.append(True)
        else:
            raise AssertionError("Empty/invalid comment must fail clearly")
    for word in ("không", "chưa", "chẳng", "chả", "đừng"):
        assert word in preprocess_text(word).split()
    example = "SV ko thích lịch học 😢 https://example.org!"
    archive_matches = {}
    archive = Path("outputs/archives/data_minimal_20261007.zip")
    if archive.is_file():
        import hashlib
        with zipfile.ZipFile(archive) as zipped:
            for path in (Path(source), *(Path(splits_dir) / (n + ".csv") for n in frames)):
                archive_matches[str(path)] = hashlib.sha256(zipped.read(path.as_posix())).hexdigest() == file_hash(path)
        assert all(archive_matches.values())
    assert before == {p: file_hash(p) for p in before}
    report = {"split_manifest_sha256": digest, "dataset_version": manifest["dataset_version"],
              "config_sha256": file_hash(config_path),
              "source_sha256": manifest["source_sha256"], "files_unchanged": before,
              "archive_byte_hash_matches": archive_matches,
              "statistics": {n: statistics(f) for n, f in frames.items()},
              "label_source_counts": original["label_source"].value_counts().to_dict(),
              "annotation_status_counts": original["annotation_status"].value_counts().to_dict(),
              "all_split_rows_equal_source": True, "split_checks_passed": True,
              "feature_checks": checks, "invalid_inputs_rejected": len(invalid),
              "example": {"raw_synthetic": example, "processed": preprocess_text(example)},
              "seconds": perf_counter() - clock}
    write_json(output, report)
    print(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--source", default="data/labeled/labeled_comments.csv")
    parser.add_argument("--config", default="configs/part3.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    review(args.splits_dir, args.source, args.config, args.output)


if __name__ == "__main__":
    run_cli(main)
