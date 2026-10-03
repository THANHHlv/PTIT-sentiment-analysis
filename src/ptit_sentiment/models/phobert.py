"""PhoBERT tùy chọn; chỉ import torch/transformers khi tải artifact."""
from pathlib import Path

from ptit_sentiment.common import read_json
from ptit_sentiment.labels import ID_TO_LABEL, LABELS, LABEL_TO_ID
from ptit_sentiment.preprocessing.tokenize import PhoBERTPreprocessor


class PhoBERTPredictor:
    """Dự đoán từ artifact local; không tải checkpoint và không huấn luyện lại."""

    def __init__(self, artifact_dir, segmenter_dir=None):
        directory = Path(artifact_dir)
        self.metadata = read_json(directory / "metadata.json")
        if self.metadata.get("label_to_id") != LABEL_TO_ID:
            raise ValueError("Ánh xạ nhãn artifact PhoBERT khác labels.py.")
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise ImportError('PhoBERT cần pip install -e ".[phobert]"') from error
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(directory, use_fast=False, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(directory, local_files_only=True)
        if self.model.config.label2id != LABEL_TO_ID:
            raise ValueError("label2id trong model config khác hợp đồng.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.preprocessor = PhoBERTPreprocessor(segmenter_dir or self.metadata["config"]["segmenter_dir"])
        self.max_length = self.metadata["config"]["max_length"]

    def predict(self, texts, batch_size=16):
        """Trả nhãn, softmax probability và thứ tự cột theo labels.py."""
        if batch_size < 1:
            raise ValueError("batch_size phải >= 1.")
        probabilities = []
        for start in range(0, len(texts), batch_size):
            segmented = self.preprocessor.transform(texts[start:start + batch_size])
            inputs = self.tokenizer(segmented, padding=True, truncation=True,
                                    max_length=self.max_length, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self.torch.no_grad():
                logits = self.model(**inputs).logits
                probabilities.extend(self.torch.softmax(logits, dim=-1).cpu().numpy())
        import numpy as np
        probabilities = np.asarray(probabilities)
        labels = [ID_TO_LABEL[int(index)] for index in probabilities.argmax(axis=1)]
        return labels, probabilities, LABELS
