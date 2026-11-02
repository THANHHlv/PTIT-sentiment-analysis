
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


# ==========================================
# 1. CONFIGURATION
# ==========================================

MODEL_NAME = "vinai/phobert-base"
MAX_LENGTH = 128

LABEL2ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

ID2LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive"
}


# ==========================================
# 2. LOAD TOKENIZER
# ==========================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False
)


# ==========================================
# 3. CREATE DATASET CLASS
# ==========================================

class PhoBERTDataset(Dataset):

    def __init__(self, csv_path, tokenizer, max_length=128):

        self.df = pd.read_csv(csv_path)

        self.tokenizer = tokenizer
        self.max_length = max_length

        # Kiểm tra dữ liệu
        required_columns = ["segmented_text", "label"]

        for column in required_columns:
            if column not in self.df.columns:
                raise ValueError(
                    f"Missing column: {column}"
                )

        # Chuẩn hóa và kiểm tra nhãn
        self.df["label"] = (
            self.df["label"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        invalid_labels = (
            set(self.df["label"].unique())
            - set(LABEL2ID.keys())
        )

        if invalid_labels:
            raise ValueError(
                f"Invalid labels: {invalid_labels}"
            )

        if self.df["segmented_text"].isna().any():
            raise ValueError("Found missing segmented_text")

        self.texts = self.df["segmented_text"].astype(str).tolist()

        self.labels = [
            LABEL2ID[label]
            for label in self.df["label"]
        ]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):

        text = self.texts[index]
        label = self.labels[index]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


# ==========================================
# 4. TEST DATASET
# ==========================================

if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    TRAIN_PATH = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "train.csv"
    )

    train_dataset = PhoBERTDataset(
        csv_path=TRAIN_PATH,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )

    print("Dataset loaded successfully!")

    print("Number of samples:", len(train_dataset))

    # Lấy thử một mẫu
    sample = train_dataset[0]

    print("\n========== SAMPLE ==========")

    print("Input IDs:")
    print(sample["input_ids"])

    print("\nAttention Mask:")
    print(sample["attention_mask"])

    print("\nLabel ID:")
    print(sample["labels"].item())

    print("\nLabel Name:")
    print(ID2LABEL[sample["labels"].item()])

    print("\nInput shape:", sample["input_ids"].shape)
    print("Attention shape:", sample["attention_mask"].shape)
