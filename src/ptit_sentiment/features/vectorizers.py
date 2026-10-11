"""Tạo vectorizer; chỉ pipeline.fit(train) được học vocabulary/IDF."""
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def whitespace_tokens(text):
    """Giữ token tách từ, emoji và dấu câu thay vì regex loại bỏ mặc định."""
    return text.split()


def make_vectorizer(kind, ngram_range=(1, 1), **kwargs):
    """BoW/TF-IDF hỗ trợ unigram đến trigram trên văn bản đã tách từ."""
    if kind not in ("bow", "tfidf"):
        raise ValueError(f"Đặc trưng không hỗ trợ: {kind}.")
    if tuple(ngram_range) not in ((1, 1), (1, 2), (1, 3)):
        raise ValueError("ngram_range phải là (1, 1), (1, 2) hoặc (1, 3).")
    cls = CountVectorizer if kind == "bow" else TfidfVectorizer
    return cls(tokenizer=whitespace_tokens, token_pattern=None, lowercase=False,
               ngram_range=tuple(ngram_range), **kwargs)


def make_tfidf_word_char(ngram_range=(1, 2), char_ngram_range=(3, 5),
                         char_max_features=20000, **word_options):
    """Ghép TF-IDF từ và ký tự trên cùng văn bản đã tách từ; fit chỉ từ train."""
    if len(char_ngram_range) != 2 or not 1 <= char_ngram_range[0] <= char_ngram_range[1]:
        raise ValueError("char_ngram_range phải là cặp số dương tăng dần.")
    return FeatureUnion([
        ("word", make_vectorizer("tfidf", ngram_range, **word_options)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(char_ngram_range),
                                  lowercase=False, min_df=word_options.get("min_df", 1),
                                  max_df=word_options.get("max_df", 1.0),
                                  max_features=char_max_features,
                                  sublinear_tf=word_options.get("sublinear_tf", False))),
    ])
