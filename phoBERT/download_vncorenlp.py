from pathlib import Path
import py_vncorenlp

# Xác định thư mục gốc project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Thư mục lưu VnCoreNLP
VNCORENLP_DIR = PROJECT_ROOT / "artifacts" / "vncorenlp"

# Tạo thư mục cha, chưa tạo models
VNCORENLP_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading VnCoreNLP...")
print("Save directory:", VNCORENLP_DIR)

try:
    py_vncorenlp.download_model(
        save_dir=str(VNCORENLP_DIR.resolve())
    )

    print("\nDownload completed!")

except Exception as e:
    print("\nDownload failed!")
    print("Error:", e)
    raise

# Kiểm tra file
jar_path = VNCORENLP_DIR / "VnCoreNLP-1.2.jar"
models_path = VNCORENLP_DIR / "models"
wordsegmenter_path = models_path / "wordsegmenter"

print("\n========== CHECK FILES ==========")
print("JAR exists:", jar_path.exists())
print("Models exists:", models_path.exists())
print("Wordsegmenter exists:", wordsegmenter_path.exists())

if wordsegmenter_path.exists():
    for file in wordsegmenter_path.iterdir():
        print(file.name, "-", file.stat().st_size, "bytes")