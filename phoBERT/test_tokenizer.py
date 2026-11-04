
from pathlib import Path
import pandas as pd

from transformers import AutoTokenizer

# ==========================================
# 1. CONFIG
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"

MODEL_NAME = "vinai/phobert-base"

MAX_LENGTH = 128

# ==========================================
# 2. LOAD TOKENIZER
# ==========================================

print("Loading PhoBERT Tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False
)

print("Tokenizer loaded successfully!")

# ==========================================
# 3. LOAD DATA
# ==========================================

df = pd.read_csv(TRAIN_PATH)

if "segmented_text" not in df.columns:
    raise ValueError("Missing segmented_text column!")

# ==========================================
# 4. TOKENIZE ONE SENTENCE
# ==========================================

text = df["segmented_text"].dropna().iloc[0]

print("\nOriginal segmented text:")
print(text)

encoding = tokenizer(
    text,
    add_special_tokens=True,
    max_length=MAX_LENGTH,
    padding="max_length",
    truncation=True,
    return_attention_mask=True,
    return_tensors="pt"
)

# ==========================================
# 5. DISPLAY TOKENIZER RESULTS
# ==========================================

input_ids = encoding["input_ids"]
attention_mask = encoding["attention_mask"]

tokens = tokenizer.convert_ids_to_tokens(
    input_ids[0]
)

print("\nTokens:")
print(tokens)

print("\nInput IDs:")
print(input_ids)

print("\nAttention Mask:")
print(attention_mask)

print("\nInput shape:", input_ids.shape)
print("Attention mask shape:", attention_mask.shape)

print("\nNumber of real tokens:", attention_mask.sum().item())

# ==========================================
# 6. DECODE
# ==========================================

decoded_text = tokenizer.decode(
    input_ids[0],
    skip_special_tokens=True
)

print("\nDecoded text:")
print(decoded_text)
