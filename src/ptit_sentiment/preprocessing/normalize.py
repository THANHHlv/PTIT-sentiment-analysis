"""Chuẩn hóa thận trọng, không làm mất thông tin phủ định."""
import re
import unicodedata

SLANG = {"ko": "không", "k": "không", "khum": "không", "hok": "không",
         "dc": "được", "đc": "được", "mn": "mọi người", "sv": "sinh viên"}
NEGATIONS = frozenset({"không", "chưa", "chẳng", "chả", "đừng"})
EMOJI_WORDS = {"😊": " vui ", "😍": " yêu thích ", "👍": " tán thành ",
               "😢": " buồn ", "😡": " tức giận ", "👎": " không thích "}
EMOJI_PATTERN = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]")


def basic_normalize(text):
    """NFC và khoảng trắng; dùng riêng trước RDRSegmenter của PhoBERT."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffd]", " ", text)
    text = text.replace("\ufeff", "").replace("\u200b", "")
    return " ".join(text.split())


def redact_sensitive(text):
    """Che email, số điện thoại VN, @handle; không suy đoán tên/người viết."""
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", " emailtoken ", text)
    text = re.sub(r"(?<!\w)(?:\+84|0)(?:[ .-]?\d){9}(?!\d)", " phonetoken ", text)
    return re.sub(r"(?<!\w)@[\w.]+", " usertoken ", text)


def normalize_text(text, lowercase=True, replace_urls=True, expand_slang=True, emoji="keep", slang_map=None, mask_sensitive=True):
    """Trả chuỗi mới; URL -> urltoken, tiếng lóng theo token, emoji cấu hình."""
    text = basic_normalize(text)
    if mask_sensitive:
        text = redact_sensitive(text)
    if lowercase:
        text = text.lower()
    if replace_urls:
        text = re.sub(r"https?://\S+|www\.\S+", " urltoken ", text, flags=re.IGNORECASE)
    if expand_slang:
        mapping = SLANG if slang_map is None else slang_map
        if not isinstance(mapping, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()):
            raise ValueError("slang_map phải là bảng chuỗi -> chuỗi.")
        text = re.sub(r"\b\w+\b", lambda match: mapping.get(match.group().lower(), match.group()), text)
    if emoji == "map":
        for symbol, word in EMOJI_WORDS.items():
            text = text.replace(symbol, word)
    elif emoji == "remove":
        text = EMOJI_PATTERN.sub(" ", text)
    elif emoji != "keep":
        raise ValueError("emoji phải là keep, map hoặc remove.")
    return " ".join(text.split())
