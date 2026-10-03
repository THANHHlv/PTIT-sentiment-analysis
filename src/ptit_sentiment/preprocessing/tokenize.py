"""Tách từ và transformer sklearn có thể lưu trong pipeline joblib."""
from sklearn.base import BaseEstimator, TransformerMixin

from ptit_sentiment.preprocessing.normalize import NEGATIONS, basic_normalize, normalize_text


def tokenize_vietnamese(text):
    """Underthesea trả chuỗi với từ nhiều âm tiết nối bằng underscore."""
    try:
        from underthesea import word_tokenize
    except ImportError as error:
        raise ImportError("Thiếu Underthesea: cài package bằng pip install -e .") from error
    return word_tokenize(text, format="text")


class TextPreprocessor(TransformerMixin, BaseEstimator):
    """Biến đổi list văn bản gốc thành list chuỗi tách từ; fit không học dữ liệu."""

    def __init__(self, lowercase=True, replace_urls=True, expand_slang=True,
                 emoji="keep", stop_words=None, tokenizer="underthesea"):
        self.lowercase = lowercase
        self.replace_urls = replace_urls
        self.expand_slang = expand_slang
        self.emoji = emoji
        self.stop_words = stop_words
        self.tokenizer = tokenizer

    def fit(self, X, y=None):
        if self.tokenizer != "underthesea":
            raise ValueError("tokenizer truyền thống phải là underthesea; không fallback theo khoảng trắng.")
        self._stop_set()
        return self

    def _stop_set(self):
        words = {word.casefold().replace(" ", "_") for word in (self.stop_words or [])}
        if any(any(part in NEGATIONS for part in word.split("_")) for word in words):
            raise ValueError("Stop words không được loại từ/cụm chứa phủ định không/chưa/chẳng/chả/đừng.")
        return words

    def transform(self, X):
        stop = self._stop_set()
        result = []
        for text in X:
            cleaned = normalize_text(text, self.lowercase, self.replace_urls, self.expand_slang, self.emoji)
            segmented = tokenize_vietnamese(cleaned)
            result.append(" ".join(token for token in segmented.split() if token.casefold() not in stop))
        return result


class PhoBERTPreprocessor:
    """NFC/khoảng trắng rồi RDRSegmenter; giữ hoa thường, emoji, phủ định."""

    def __init__(self, segmenter_dir):
        from pathlib import Path
        directory = Path(segmenter_dir).resolve()
        required = ("VnCoreNLP-1.2.jar", "models/wordsegmenter/vi-vocab",
                    "models/wordsegmenter/wordsegmenter.rdr")
        if any(not (directory / name).is_file() for name in required):
            raise FileNotFoundError(f"Thiếu VnCoreNLP tại {directory}. Chạy scripts/setup_vncorenlp.py; xem README.")
        try:
            import py_vncorenlp
        except ImportError as error:
            raise ImportError('Cài phụ thuộc riêng: pip install -e ".[phobert]"') from error
        # Wrapper đổi cwd khi khởi tạo JVM; phục hồi để không làm lệch đường dẫn CLI.
        import os
        previous_directory = os.getcwd()
        try:
            self.segmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=str(directory))
        except Exception as error:
            raise ValueError(f"Không khởi tạo được VnCoreNLP: {error}. Kiểm tra Java/JAVA_HOME và model local.") from error
        finally:
            os.chdir(previous_directory)

    def transform(self, texts):
        """Trả một chuỗi tách từ cho mỗi bình luận, nối các câu trong bình luận."""
        return [" ".join(self.segmenter.word_segment(basic_normalize(text))) for text in texts]
