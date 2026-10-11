"""Public preprocessing API for classical training and prediction.

The implementation is shared with the saved sklearn pipeline. Importing this
module does not load Underthesea until word segmentation is requested.
"""

from ptit_sentiment.preprocessing.normalize import basic_normalize, normalize_text
from ptit_sentiment.preprocessing.tokenize import TextPreprocessor, tokenize_vietnamese


def preprocess_text(text: str, **options) -> str:
    """Normalize and segment one raw comment using TextPreprocessor options.

    Args:
        text: Original Unicode comment. It is never changed in place.
        **options: Settings such as emoji, stop_words and slang_map from the
            preprocessing section of configs/classical.yaml.

    Returns:
        A normalized, Vietnamese word-segmented string for BoW or TF-IDF.

    This helper has no fitted state. Saved model pipelines use TextPreprocessor
    directly so training and prediction share the same implementation.
    """
    if not isinstance(text, str):
        raise TypeError("text phải là chuỗi Unicode.")
    if not text.strip():
        raise ValueError("Bình luận không được rỗng.")
    return TextPreprocessor(**options).fit([]).transform([text])[0]


__all__ = [
    "basic_normalize", "normalize_text", "tokenize_vietnamese",
    "TextPreprocessor", "preprocess_text",
]