
import os
from pathlib import Path

# ==========================================
# 1. CẤU HÌNH JAVA
# ==========================================

JAVA_HOME = r"C:\Program Files\Eclipse Adoptium\jdk-11.0.31.11-hotspot"

os.environ["JAVA_HOME"] = JAVA_HOME
os.environ["PATH"] = (
    os.path.join(JAVA_HOME, "bin")
    + os.pathsep
    + os.environ.get("PATH", "")
)

# Import sau khi cấu hình Java
import pandas as pd
import py_vncorenlp

# ==========================================
# 2. XÁC ĐỊNH ĐƯỜNG DẪN
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "splits"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

VNCORENLP_DIR = PROJECT_ROOT / "artifacts" / "vncorenlp"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "train": DATA_DIR / "train.csv",
    "validation": DATA_DIR / "validation.csv",
    "test": DATA_DIR / "test.csv"
}

# ==========================================
# 3. KHỞI TẠO WORD SEGMENTER
# ==========================================

print("Initializing VnCoreNLP...")

segmenter = py_vncorenlp.VnCoreNLP(
    annotators=["wseg"],
    save_dir=str(VNCORENLP_DIR.resolve())
)

print("VnCoreNLP initialized successfully!")

# ==========================================
# 4. HÀM TIỀN XỬ LÝ
# ==========================================

def preprocess_text(text):
    """
    Tách từ tiếng Việt bằng VnCoreNLP.
    Giữ nguyên dấu tiếng Việt và từ mang ý nghĩa cảm xúc.
    """

    if pd.isna(text):
        return ""

    text = str(text).strip()

    if not text:
        return ""

    segmented = segmenter.word_segment(text)

    return " ".join(segmented)


# ==========================================
# 5. XỬ LÝ 3 DATASET
# ==========================================

for split_name, file_path in FILES.items():

    print(f"\nProcessing {split_name}...")

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)

    # Kiểm tra cột đầu vào
    if "text" not in df.columns:
        raise ValueError(f"Missing 'text' column in {file_path}")

    if "label" not in df.columns:
        raise ValueError(f"Missing 'label' column in {file_path}")

    # Tách từ
    df["segmented_text"] = df["text"].apply(preprocess_text)

    # Lưu dữ liệu
    output_path = OUTPUT_DIR / f"{split_name}.csv"

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("Input rows:", len(df))
    print("Output:", output_path)

    # Hiển thị ví dụ
    print("\nExamples:")

    for _, row in df[["text", "segmented_text", "label"]].head(3).iterrows():
        print("Original:", row["text"])
        print("Segmented:", row["segmented_text"])
        print("Label:", row["label"])
        print("-" * 50)

print("\nAll datasets processed successfully!")
