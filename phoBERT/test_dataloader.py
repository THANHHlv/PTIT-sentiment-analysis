
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import PhoBERTDataset


# ==========================================
# 1. CONFIG
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"

MODEL_NAME = "vinai/phobert-base"

MAX_LENGTH = 128
BATCH_SIZE = 8


# ==========================================
# 2. LOAD TOKENIZER
# ==========================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False
)


# ==========================================
# 3. CREATE DATASET
# ==========================================

train_dataset = PhoBERTDataset(
    csv_path=TRAIN_PATH,
    tokenizer=tokenizer,
    max_length=MAX_LENGTH
)


# ==========================================
# 4. CREATE DATALOADER
# ==========================================

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


# ==========================================
# 5. TEST ONE BATCH
# ==========================================

batch = next(iter(train_loader))

print("========== DATALOADER TEST ==========")

print("Input IDs shape:", batch["input_ids"].shape)

print("Attention Mask shape:", batch["attention_mask"].shape)

print("Labels shape:", batch["labels"].shape)

print("\nLabels:", batch["labels"])

print("\nGPU available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
