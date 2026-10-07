"""Tạo vectorizer; chỉ pipeline.fit(train) được học vocabulary/IDF."""
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def whitespace_tokens(text):
    """Giữ token tách từ, emoji và dấu câu thay vì regex loại bỏ mặc định."""
    return text.split()


def make_vectorizer(kind, ngram_range=(1, 1), **kwargs):
    """BoW/TF-IDF hỗ trợ unigram/bigram; input đã được chuẩn hóa/tách từ."""
    if kind not in ("bow", "tfidf"):
        raise ValueError(f"Đặc trưng không hỗ trợ: {kind}.")
    if tuple(ngram_range) not in ((1, 1), (1, 2)):
        raise ValueError("ngram_range phải là (1, 1) hoặc (1, 2).")
    cls = CountVectorizer if kind == "bow" else TfidfVectorizer
    return cls(tokenizer=whitespace_tokens, token_pattern=None, lowercase=False,
               ngram_range=tuple(ngram_range), **kwargs)
