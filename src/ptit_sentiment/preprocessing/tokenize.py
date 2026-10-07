"""Tách từ và transformer sklearn có thể lưu trong pipeline joblib."""
from sklearn.base import BaseEstimator, TransformerMixin

from ptit_sentiment.preprocessing.normalize import NEGATIONS, basic_normalize, normalize_text, redact_sensitive


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
                 emoji="keep", stop_words=None, tokenizer="underthesea", slang_map=None, mask_sensitive=True):
        self.lowercase = lowercase
        self.replace_urls = replace_urls
        self.expand_slang = expand_slang
        self.emoji = emoji
        self.stop_words = stop_words
        self.tokenizer = tokenizer
        self.slang_map = slang_map
        self.mask_sensitive = mask_sensitive

    def fit(self, X, y=None):
        if self.tokenizer != "underthesea":
            raise ValueError("tokenizer truyền thống phải là underthesea; không fallback theo khoảng trắng.")
        self._stop_set()
        return self

    def _stop_set(self):
        if self.stop_words is not None and (
            not isinstance(self.stop_words, (list, tuple, set))
            or any(not isinstance(word, str) for word in self.stop_words)
        ):
            raise ValueError("stop_words phải là danh sách các chuỗi.")
        words = {word.casefold().replace(" ", "_") for word in (self.stop_words or [])}
        if any(any(part in NEGATIONS for part in word.split("_")) for word in words):
            raise ValueError("Stop words không được loại từ/cụm chứa phủ định không/chưa/chẳng/chả/đừng.")
        return words

    def transform(self, X):
        stop = self._stop_set()
        result = []
        for text in X:
            cleaned = normalize_text(text, self.lowercase, self.replace_urls, self.expand_slang, self.emoji,
                                     getattr(self, "slang_map", None), getattr(self, "mask_sensitive", True))
            segmented = tokenize_vietnamese(cleaned)
            result.append(" ".join(token for token in segmented.split() if token.casefold() not in stop))
        return result


class PhoBERTPreprocessor:
    """NFC/khoảng trắng rồi RDRSegmenter; giữ hoa thường, emoji, phủ định."""

    def __init__(self, segmenter_dir, mask_sensitive=True):
        self.mask_sensitive = mask_sensitive
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
            self.use_fallback = False
        except Exception as error:
            # Fallback sang Underthesea nếu môi trường máy người dùng chưa có JAVA_HOME
            self.segmenter = None
            self.use_fallback = True
        finally:
            os.chdir(previous_directory)

    def transform(self, texts):
        """Trả một chuỗi tách từ cho mỗi bình luận, nối các câu trong bình luận."""
        cleaned = [basic_normalize(text) for text in texts]
        if self.mask_sensitive:
            cleaned = [redact_sensitive(text) for text in cleaned]
        if getattr(self, "use_fallback", False) or self.segmenter is None:
            return [tokenize_vietnamese(text) for text in cleaned]
        return [" ".join(self.segmenter.word_segment(text)) for text in cleaned]
