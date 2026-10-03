"""Nguồn duy nhất cho thứ tự và ánh xạ ba nhãn."""

LABELS = ("positive", "neutral", "negative")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}
