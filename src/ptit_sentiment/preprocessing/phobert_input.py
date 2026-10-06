"""Token hóa PhoBERT và thống kê truncation bằng tokenizer thực khi có model."""


def encode_with_statistics(tokenizer, texts, max_length, **kwargs):
    """Trả encoding và thống kê độ dài trước cắt (kể cả special tokens).

    Cắt bên phải theo tokenizer.truncation_side, giữ tối đa max_length token.
    Chỉ lưu vị trí/độ dài, không chép nội dung vào log.
    """
    lengths = [len(ids) for ids in tokenizer(texts, truncation=False)["input_ids"]]
    stats = {
        "samples": len(texts), "max_length": max_length,
        "truncation_side": tokenizer.truncation_side,
        "truncated_count": sum(length > max_length for length in lengths),
        "truncated_samples": [{"index": i, "original_tokens": length}
                              for i, length in enumerate(lengths) if length > max_length],
        "max_original_tokens": max(lengths, default=0),
    }
    stats["truncated_fraction"] = stats["truncated_count"] / len(texts) if texts else 0
    return tokenizer(texts, truncation=True, max_length=max_length, **kwargs), stats
