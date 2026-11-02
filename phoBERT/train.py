
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score
)

from tqdm import tqdm

from dataset import (
    PhoBERTDataset,
    LABEL2ID,
    ID2LABEL
)


# ==================================================
# 1. CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
VAL_PATH = PROJECT_ROOT / "data" / "processed" / "validation.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phobert" / "best_model"

MODEL_NAME = "vinai/phobert-base"

MAX_LENGTH = 128

BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4

EPOCHS = 5

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

WARMUP_RATIO = 0.1

RANDOM_SEED = 42

NUM_LABELS = 3


# ==================================================
# 2. SET RANDOM SEED
# ==================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_SEED)


# ==================================================
# 3. DEVICE
# ==================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_FP16 = DEVICE.type == "cuda"

print("=" * 60)
print("DEVICE:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

print("FP16:", USE_FP16)
print("=" * 60)


# ==================================================
# 4. CHECK DATA
# ==================================================

if not TRAIN_PATH.exists():
    raise FileNotFoundError(f"Train file not found: {TRAIN_PATH}")

if not VAL_PATH.exists():
    raise FileNotFoundError(f"Validation file not found: {VAL_PATH}")


# ==================================================
# 5. LOAD TOKENIZER
# ==================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False
)


# ==================================================
# 6. CREATE DATASETS
# ==================================================

print("\nLoading datasets...")

train_dataset = PhoBERTDataset(
    csv_path=TRAIN_PATH,
    tokenizer=tokenizer,
    max_length=MAX_LENGTH
)

val_dataset = PhoBERTDataset(
    csv_path=VAL_PATH,
    tokenizer=tokenizer,
    max_length=MAX_LENGTH
)

print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


# ==================================================
# 7. CREATE DATALOADERS
# ==================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ==================================================
# 8. LOAD PHOBERT MODEL
# ==================================================

print("\nLoading PhoBERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=ID2LABEL,
    label2id=LABEL2ID
)

model.to(DEVICE)


# ==================================================
# 9. OPTIMIZER
# ==================================================

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

total_optimizer_steps = (
    (len(train_loader) + GRADIENT_ACCUMULATION_STEPS - 1)
    // GRADIENT_ACCUMULATION_STEPS
) * EPOCHS

warmup_steps = int(total_optimizer_steps * WARMUP_RATIO)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_optimizer_steps
)

scaler = torch.cuda.amp.GradScaler(
    enabled=USE_FP16
)


# ==================================================
# 10. VALIDATION FUNCTION
# ==================================================

def evaluate(model, dataloader):

    model.eval()

    all_predictions = []
    all_labels = []

    total_loss = 0.0

    with torch.no_grad():

        for batch in tqdm(dataloader, desc="Validation"):

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            with torch.cuda.amp.autocast(enabled=USE_FP16):

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                logits = outputs.logits

            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    avg_loss = total_loss / max(len(dataloader), 1)

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision, recall, class_f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        labels=[0, 1, 2],
        zero_division=0
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0
    )

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "negative_precision": precision[0],
        "negative_recall": recall[0],
        "negative_f1": class_f1[0],
        "neutral_precision": precision[1],
        "neutral_recall": recall[1],
        "neutral_f1": class_f1[1],
        "positive_precision": precision[2],
        "positive_recall": recall[2],
        "positive_f1": class_f1[2]
    }

    return metrics


# ==================================================
# 11. TRAINING
# ==================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

best_macro_f1 = -1.0
history = []

print("\nStarting fine-tuning...")

for epoch in range(EPOCHS):

    model.train()

    total_train_loss = 0.0

    optimizer.zero_grad(set_to_none=True)

    progress_bar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for step, batch in progress_bar:

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        # Tính số batch thực tế trong nhóm tích lũy gradient
        group_start = (
            step // GRADIENT_ACCUMULATION_STEPS
        ) * GRADIENT_ACCUMULATION_STEPS

        group_size = min(
            GRADIENT_ACCUMULATION_STEPS,
            len(train_loader) - group_start
        )

        with torch.cuda.amp.autocast(enabled=USE_FP16):

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss

            # Gradient accumulation
            scaled_loss = loss / group_size

        scaler.scale(scaled_loss).backward()

        total_train_loss += loss.item()

        should_update = (
            (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0
            or (step + 1) == len(train_loader)
        )

        if should_update:

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            optimizer.zero_grad(set_to_none=True)

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}"
        })

    avg_train_loss = total_train_loss / max(
        len(train_loader), 1
    )

    # ==============================================
    # 12. VALIDATE AFTER EACH EPOCH
    # ==============================================

    val_metrics = evaluate(model, val_loader)

    print("\n" + "=" * 60)
    print(f"EPOCH {epoch + 1}/{EPOCHS}")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Validation Loss: {val_metrics['loss']:.4f}")
    print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Validation Macro-F1: {val_metrics['macro_f1']:.4f}")

    print("\nPer-class F1:")
    print(f"Negative: {val_metrics['negative_f1']:.4f}")
    print(f"Neutral:  {val_metrics['neutral_f1']:.4f}")
    print(f"Positive: {val_metrics['positive_f1']:.4f}")

    print("=" * 60)

    # Lưu lịch sử
    epoch_result = {
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        **val_metrics
    }

    history.append(epoch_result)

    # ==============================================
    # 13. SAVE BEST MODEL
    # ==============================================

    if val_metrics["macro_f1"] > best_macro_f1:

        best_macro_f1 = val_metrics["macro_f1"]

        print("\nNew best model! Saving...")

        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        config = {
            "model_name": MODEL_NAME,
            "max_length": MAX_LENGTH,
            "label2id": LABEL2ID,
            "id2label": ID2LABEL,
            "best_validation_macro_f1": best_macro_f1
        }

        with open(
            OUTPUT_DIR / "training_config.json",
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                config,
                file,
                ensure_ascii=False,
                indent=4
            )

        print("Best model saved at:", OUTPUT_DIR)


# ==================================================
# 14. SAVE TRAINING HISTORY
# ==================================================

history_path = (
    PROJECT_ROOT
    / "outputs"
    / "phobert"
    / "training_history.csv"
)

pd.DataFrame(history).to_csv(
    history_path,
    index=False,
    encoding="utf-8-sig"
)

print("\nTraining completed!")
print("Best Validation Macro-F1:", best_macro_f1)
print("Model directory:", OUTPUT_DIR)
print("Training history:", history_path)
