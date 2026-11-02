
import os
from pathlib import Path

# 1. Cấu hình JAVA_HOME trước khi import py_vncorenlp
JAVA_HOME = r"C:\Program Files\Eclipse Adoptium\jdk-11.0.31.11-hotspot"

os.environ["JAVA_HOME"] = JAVA_HOME
os.environ["PATH"] = os.path.join(JAVA_HOME, "bin") + os.pathsep + os.environ.get("PATH", "")

print("JAVA_HOME:", os.environ.get("JAVA_HOME"))

# 2. Sau khi cấu hình Java mới import thư viện
import py_vncorenlp

# 3. Xác định thư mục project
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VNCORENLP_DIR = PROJECT_ROOT / "artifacts" / "vncorenlp"

# 4. Khởi tạo VnCoreNLP
print("Initializing VnCoreNLP...")

segmenter = py_vncorenlp.VnCoreNLP(
    annotators=["wseg"],
    save_dir=str(VNCORENLP_DIR.resolve())
)

# 5. Kiểm tra tách từ
sentences = [
    "Tôi rất thích học máy.",
    "Trường PTIT có cơ sở vật chất rất tốt.",
    "Hệ thống đăng ký tín chỉ quá tệ.",
    "Không sao nhé, chậm cả tháng cũng được.",
    "Môn xử lý ngôn ngữ tự nhiên rất thú vị."
]

for text in sentences:
    result = segmenter.word_segment(text)

    print("\nOriginal: ", text)
    print("Segmented:", " ".join(result))
    print("-" * 60)
