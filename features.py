"""NV2 handoff: fit BoW or TF-IDF on train, reuse it everywhere else.

The vectorizer implementation remains in ptit_sentiment.features. This module
shows the complete feature extraction boundary for the three handed-off splits.
It does not train a classifier, read labels, or change the source CSV files.
"""

from dataclasses import dataclass
from scipy.sparse import csr_matrix

from ptit_sentiment.features import make_vectorizer
from ptit_sentiment.preprocess import TextPreprocessor


@dataclass
class FeatureBundle:
    """Fitted preprocessing, vectorizer and sparse matrices for three splits."""

    preprocessor: TextPreprocessor
    vectorizer: object
    X_train: object
    X_validation: object
    X_test: object

    def transform_comment(self, text: str):
        """Return a 1-row sparse vector for a new raw comment; never refit."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Bình luận mới phải là chuỗi không rỗng.")
        processed = self.preprocessor.transform([text])
        return self.vectorizer.transform(processed)


def build_feature_matrices(
    train_texts,
    validation_texts,
    test_texts,
    *,
    kind="tfidf",
    ngram_range=(1, 2),
    preprocessing_options=None,
    vectorizer_options=None,
):
    """Fit only on raw train texts; transform validation and test with that fit.

    Args:
        train_texts, validation_texts, test_texts: Iterables of raw comments.
        kind: ``bow`` or ``tfidf``.
        ngram_range: ``(1, 1)`` or ``(1, 2)``.
        preprocessing_options: Settings from ``classical.yaml:preprocessing``.
        vectorizer_options: Settings from ``classical.yaml:vectorizer``.

    Returns:
        FeatureBundle with fitted objects and three sparse matrices. The same
        objects also transform a new comment through ``transform_comment``.
    """
    train_texts = list(train_texts)
    validation_texts = list(validation_texts)
    test_texts = list(test_texts)
    if not train_texts:
        raise ValueError("Train rỗng; không thể học vocabulary.")
    preprocessor = TextPreprocessor(**(preprocessing_options or {}))
    train_tokens = preprocessor.fit_transform(train_texts)
    vectorizer = make_vectorizer(kind, ngram_range, **(vectorizer_options or {}))
    X_train = vectorizer.fit_transform(train_tokens)
    def transform_optional(texts):
        if not texts:
            return csr_matrix((0, X_train.shape[1]), dtype=X_train.dtype)
        return vectorizer.transform(preprocessor.transform(texts))

    X_validation = transform_optional(validation_texts)
    X_test = transform_optional(test_texts)
    return FeatureBundle(preprocessor, vectorizer, X_train, X_validation, X_test)