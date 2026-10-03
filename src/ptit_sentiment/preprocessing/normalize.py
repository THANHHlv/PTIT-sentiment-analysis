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
    return " ".join(unicodedata.normalize("NFC", text).split())


def normalize_text(text, lowercase=True, replace_urls=True, expand_slang=True, emoji="keep"):
    """Trả chuỗi mới; URL -> urltoken, tiếng lóng theo token, emoji cấu hình."""
    text = basic_normalize(text)
    if lowercase:
        text = text.lower()
    if replace_urls:
        text = re.sub(r"https?://\S+|www\.\S+", " urltoken ", text, flags=re.IGNORECASE)
    if expand_slang:
        text = re.sub(r"\b\w+\b", lambda match: SLANG.get(match.group().lower(), match.group()), text)
    if emoji == "map":
        for symbol, word in EMOJI_WORDS.items():
            text = text.replace(symbol, word)
    elif emoji == "remove":
        text = EMOJI_PATTERN.sub(" ", text)
    elif emoji != "keep":
        raise ValueError("emoji phải là keep, map hoặc remove.")
    return " ".join(text.split())
