"""Tạo bản bàn giao có chọn lọc: code GitHub, data nội bộ, hướng dẫn bốn người.

Chỉ sao chép/kiểm tra dữ liệu hiện có; không thu thập, fit hoặc train.
Đầu ra nằm trong handoff/ và bị Git bỏ qua; từ chối ghi đè bản bàn giao.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from ptit_sentiment.common import file_hash, read_json, run_cli, write_json
from ptit_sentiment.data.split import load_split_set
from ptit_sentiment.data.annotation_review import read_table
from package_team_data import package_data

PUBLIC_SCRIPTS = (
    'run_smoke.py', 'run_phobert_smoke.py', 'setup_vncorenlp.py',
    'download_phobert.py', 'package_team_data.py', 'prepare_team_handoff.py',
    'export_report_evidence.py', 'verify_ai_dataset.py', 'verify_annotation_run.py',
)
PUBLIC_DOCS = (
    'data_contract.md', 'labeling_guide.md', 'annotation_review.md',
    'methodology.md', 'learning_guide.md', 'task_assignment.md', 'report_outline.md',
    'team_handoff.md',
)
PUBLIC_CONFIGS = ('classical.yaml', 'phobert.yaml', 'phobert_gpu4gb.yaml', 'phobert_smoke.yaml')

CODE_README = r'''# PTIT Sentiment Analysis — bản code bàn giao

Đề tài: Xây dựng hệ thống phân loại cảm xúc bình luận của sinh viên PTIT trên các nhóm Facebook.

Code dùng chung cho bốn người: BoW/TF-IDF × MultinomialNB/LinearSVC,
baseline lớp phổ biến nhất và fine-tune PhoBERT. Đây là bản sao để bàn giao
của dự án hiện có; tiếp tục làm việc trên cùng kho và package ptit_sentiment.

## Trạng thái dữ liệu và thực nghiệm

Bộ AI ban đầu: 2.600 dòng có nhãn (positive 362, neutral 1.800, negative 438),
194 unresolved, 70 bị loại từ 2.864 dòng gốc. Chưa có nhãn người xác nhận.
Train/validation/test: 1.801/409/390; version ai_initial_4e93595ef7d1_seed42.
Nhãn AI giữ label_source=ai và annotation_status=ai_labeled.
Test AI chỉ đo mức phù hợp với nhãn AI, chưa thay thế nhãn người độc lập.
Không có kết quả huấn luyện mới trên bộ này trong lần đóng gói.

## Cài đặt và nhận dữ liệu

Python 3.10–3.12, PowerShell tại gốc kho:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

Nhận folder 02_DATA_PRIVATE riêng từ người 1. Sao chép thư mục data và
TEAM_DATA.json bên trong nó vào gốc clone của bạn. tests/fixtures có sẵn
trong GitHub là dữ liệu tổng hợp dùng thử code; không dùng làm thực nghiệm thật.

```powershell
.\.venv\Scripts\python.exe -m ptit_sentiment.data.validate --input data/labeled/labeled_comments.csv
.\.venv\Scripts\python.exe -c "from ptit_sentiment.data.split import load_split_set; f,m,h=load_split_set('data/splits'); print(m['dataset_version'], h, {k:len(v) for k,v in f.items()})"
```

Dùng cùng bộ chia cho mọi mô hình. Không sửa manifest, không chia lại trên
từng máy. Nhãn mới phải tạo phiên bản dữ liệu mới cho cả nhóm.

## Lệnh thực nghiệm cho bước tiếp theo

Các lệnh train/evaluate dưới đây chưa được chạy trong lần đóng gói này.
Train chỉ học train, chọn cấu hình bằng Macro-F1 validation; chốt cấu hình
trước khi chạy evaluate test.

```powershell
$python = ".\.venv\Scripts\python.exe"
& $python -m ptit_sentiment.training.train_classical --splits-dir data/splits --config configs/classical.yaml --artifact-dir artifacts/ai_initial/classical --output-dir outputs/ai_initial/training
$models = @("bow_nb", "tfidf_nb", "bow_svm", "tfidf_svm", "majority_baseline")
foreach ($model in $models) {
  & $python -m ptit_sentiment.evaluation.evaluate --model "artifacts/ai_initial/classical/$model.joblib" --splits-dir data/splits --output-dir "outputs/ai_initial/evaluation/$model"
}
$metricPaths = $models | ForEach-Object { "outputs/ai_initial/evaluation/$_/metrics.json" }
& $python -m ptit_sentiment.evaluation.compare --inputs $metricPaths --output outputs/ai_initial/comparison.csv
& $python -m ptit_sentiment.predict --model artifacts/ai_initial/classical/tfidf_svm.joblib --text "Mình hài lòng với lịch học mới."
& $python -m ptit_sentiment.predict --model artifacts/ai_initial/classical/tfidf_svm.joblib --input tests/fixtures/new_comments.csv --output outputs/ai_initial/new_predictions.csv
```

PhoBERT cài riêng; tải checkpoint công khai, không gửi bình luận:

```powershell
& $python -m pip install -e ".[phobert]"
java -version
& $python scripts/setup_vncorenlp.py
& $python scripts/download_phobert.py
& $python -m ptit_sentiment.training.check_phobert --config configs/phobert_gpu4gb.yaml --output outputs/ai_initial/phobert_preflight.json
& $python -m ptit_sentiment.training.train_phobert --splits-dir data/splits --config configs/phobert_gpu4gb.yaml --artifact-dir artifacts/ai_initial/phobert
# Chốt checkpoint bằng validation rồi mới chạy test:
& $python -m ptit_sentiment.evaluation.evaluate --model artifacts/ai_initial/phobert --splits-dir data/splits --output-dir outputs/ai_initial/evaluation/phobert
```

Tên output/artifact phải chưa tồn tại. Cấu hình smoke chỉ kiểm tra tối thiểu,
không thay fine-tune đầy đủ. Chỉ so sánh những lần đánh giá cùng manifest/test.

## Phân công và báo cáo

Người 1: data/validate/split/gán nhãn. Người 2: preprocessing/features.
Người 3: models/classical, training/train_classical, evaluation.
Người 4: PhoBERT/predict/tích hợp. Xem docs/task_assignment.md và docs/learning_guide.md.
Tất cả cùng rà nhãn bằng annotation_batches_ai_v1; người 1 ghép kết quả.

Báo cáo cũ và script giả xác nhận không nằm trong bản này. report/README.md
ghi trạng thái cần viết báo cáo; docs/report_outline.md cung cấp đề cương.
Không ghi Kappa/đồng thuận người hoặc kết luận mô hình tốt nhất từ nhãn cũ.
Chưa kiểm chứng cài mới từ Internet trên máy thành viên khác.

## Những gì đưa lên GitHub

Đưa nội dung folder 01_CODE_GITHUB lên gốc kho, giữ src/configs/docs/tests/scripts.
Không upload toàn bộ folder bàn giao. 02_DATA_PRIVATE chứa text gốc và nguồn,
chỉ chia sẻ trong nhóm được phép sử dụng; không upload vào kho Public/Release.
Không commit dữ liệu thật, cookie/profile Chrome, cache, model, output lớn.
Sau khi commit code, mỗi người làm nhánh riêng và gửi PR vào cùng kho.
'''


def copy_file(root, code, name):
    """Sao chép đúng một file được chọn; trả đường dẫn tương đối sản phẩm."""
    source, target = root / name, code / name
    if not source.is_file():
        raise FileNotFoundError(f'Thiếu file code: {name}')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def write_public_index(code):
    """Viết hướng dẫn tự chứa và danh sách file được phép upload."""
    guide = code / 'docs/team_handoff.md'
    content = guide.read_text(encoding='utf-8').partition('Xem thêm [chính sách GitHub')[0]
    guide.write_text(content + 'Xem README.md tại gốc code cho chính sách chia sẻ và lệnh sử dụng.\n',
                     encoding='utf-8')
    names = sorted(p.relative_to(code).as_posix() for p in code.rglob('*')
                   if p.is_file() and p.name != 'FILES_TO_UPLOAD.txt')
    (code / 'FILES_TO_UPLOAD.txt').write_text(
        'Chỉ đưa các file sau lên gốc kho; không upload 02_DATA_PRIVATE.\n\n'
        + '\n'.join(names) + '\n', encoding='utf-8')


def prepare(root, destination):
    """Tạo folder mới, kiểm tra hash dữ liệu và phân công, lưu manifest bàn giao."""
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if not destination.is_relative_to(root / 'handoff'):
        raise ValueError('Đầu ra phải nằm trong handoff/ để tránh thêm nhầm vào Git.')
    if destination.exists():
        raise ValueError('Folder bàn giao đã có; chọn tên mới, không ghi đè.')
    _, manifest, digest = load_split_set(root / 'data/splits')
    if manifest.get('storage_layout') == 'compact_v1':
        raise ValueError('Đã thu gọn data. Dùng package_team_data.py cho bộ hiện tại; '
                         'muốn bàn giao phân công người, khôi phục hồ sơ từ outputs/archives/data_minimal_20261007.zip trước.')
    if manifest['dataset_version'] != 'ai_initial_4e93595ef7d1_seed42':
        raise ValueError('README mẫu dành cho bộ AI hiện hành; cập nhật mẫu cho version mới trước khi đóng gói.')
    assignment_source = root / 'data/labeled/annotation_batches_ai_v1'
    assignment = read_json(assignment_source / 'assignment_manifest.json')
    clean = root / 'data/labeled/comments_clean.csv'
    if file_hash(clean) != assignment['data_sha256']:
        raise ValueError('Dữ liệu và phân công rà nhãn không cùng phiên bản.')
    if file_hash(assignment_source / 'base.csv') != assignment['base_sha256']:
        raise ValueError('Bản nền phân công đã thay đổi.')
    clean_rows, _ = read_table(clean)
    all_rows, _ = read_table(root / 'data/labeled/labeled_comments.csv')
    unresolved, _ = read_table(root / 'data/labeled/unresolved.csv')
    if {r['id'] for r in clean_rows} != {r['id'] for r in all_rows + unresolved}:
        raise ValueError('Danh sách rà nhãn khác labeled + unresolved hiện hành.')
    destination.mkdir(parents=True)
    code = destination / '01_CODE_GITHUB'
    private = destination / '02_DATA_PRIVATE'
    guides = destination / '03_HUONG_DAN_THANH_VIEN'
    code.mkdir(); private.mkdir(); guides.mkdir()
    for name in ('AGENTS.md', '.gitignore', 'pyproject.toml'):
        copy_file(root, code, name)
    for directory, pattern in (('src', '*.py'), ('tests', '*.py')):
        for source in sorted((root / directory).rglob(pattern)):
            copy_file(root, code, source.relative_to(root).as_posix())
    for directory, names in (('configs', PUBLIC_CONFIGS), ('docs', PUBLIC_DOCS), ('scripts', PUBLIC_SCRIPTS)):
        for name in names:
            copy_file(root, code, f'{directory}/{name}')
    for source in sorted((root / 'tests/fixtures').iterdir()):
        if source.is_file() and source.suffix in ('.csv', '.md'):
            copy_file(root, code, source.relative_to(root).as_posix())
    for name in ('data/labeled/template.csv', 'data/raw/.gitkeep', 'data/labeled/.gitkeep',
                 'data/splits/.gitkeep', 'artifacts/.gitkeep', 'outputs/.gitkeep'):
        copy_file(root, code, name)
    (code / 'README.md').write_text(CODE_README.replace('\\\\', '\\'), encoding='utf-8')
    (code / 'report').mkdir()
    (code / 'report/README.md').write_text(
        '# Trạng thái báo cáo\n\nDùng docs/report_outline.md để viết báo cáo mới. '
        'Bộ hiện tại là nhãn AI ban đầu, chưa có nhãn người xác nhận, chưa train/evaluate '
        'lại trong lần bàn giao. Không sử dụng khẳng định hoàn thành, Kappa hoặc '
        'kết quả test của bản cũ làm kết quả hiện hành.\n', encoding='utf-8')
    write_public_index(code)
    with TemporaryDirectory(prefix='package_', dir=destination) as scratch:
        archive, summary = package_data(root, Path(scratch) / 'data.zip')
        with ZipFile(archive) as stream:
            stream.extractall(private)
    for name in ('comments_clean.csv', 'review_sample.csv'):
        target = private / 'data/labeled' / name
        shutil.copy2(root / 'data/labeled' / name, target)
    assignments = private / 'data/labeled/annotation_batches_ai_v1'
    assignments.mkdir()
    for name in ('base.csv', 'common_sample.csv', *(f'annotator_{n}.csv' for n in range(1, 5))):
        shutil.copy2(assignment_source / name, assignments / name)
    # Chỉ thay đường dẫn cấu hình trong bản sao; CSV, ID, nhãn và manifest split giữ nguyên.
    assignment['clean_file'] = 'data/labeled/comments_clean.csv'
    write_json(assignments / 'assignment_manifest.json', assignment)
    summary['files_sha256'] = {
        p.relative_to(private).as_posix(): file_hash(p)
        for p in sorted((private / 'data').rglob('*')) if p.is_file()}
    summary['assignment_source_manifest_sha256'] = file_hash(assignment_source / 'assignment_manifest.json')
    summary['assignment_portability_note'] = 'Only clean_file changed to relative path in copied assignment manifest.'
    write_json(private / 'TEAM_DATA.json', summary)
    (private / 'README.md').write_text(
        '# Dữ liệu nội bộ\n\nChỉ chia sẻ với bốn thành viên được phép sử dụng. '
        'Chứa text nguyên văn và nguồn, chưa ẩn danh để công khai. Sao chép data/ '
        'và TEAM_DATA.json vào gốc clone code; chạy validate và load_split_set theo '
        'README code. Bộ AI 2.600 dòng; 194 unresolved, 70 bị loại. '
        'annotation_batches_ai_v1 là đợt rà người mới, label đang trống/pending. '
        'Không sửa CSV trong splits; tạo version mới nếu nhãn thay đổi.\n', encoding='utf-8')
    tasks = {
        1: ('Dữ liệu và điều phối', 'src/ptit_sentiment/data/',
            'Kiểm tra nguồn/ID/trùng, điều phối annotation_batches_ai_v1, giải quyết unresolved; '
            'nhận các file thành viên, merge ra thư mục mới; phát hành version cho cả nhóm. '
            'Raw/snapshot/checkpoint vẫn giữ tại máy gốc, không có trong gói.'),
        2: ('Tiền xử lý và đặc trưng', 'src/ptit_sentiment/preprocessing/ và src/ptit_sentiment/features/',
            'Hoàn thiện chuẩn hóa/tách từ/BoW/TF-IDF. Giữ phủ định; fit vocabulary/IDF trên train. '
            'Bàn giao transformer tải/lưu được cho người 3; phối hợp input PhoBERT với người 4.'),
        3: ('NB, SVM và đánh giá', 'src/ptit_sentiment/models/classical.py, training/train_classical.py, evaluation/',
            'Chạy bốn tổ hợp và baseline sau khi chốt preprocessing. Chọn tham số bằng '
            'validation Macro-F1; chốt cấu hình rồi mới test. Lưu artifact/output riêng, '
            'so sánh đúng manifest và ghi rõ test nhãn AI.'),
        4: ('PhoBERT và tích hợp', 'src/ptit_sentiment/models/phobert.py, training/train_phobert.py, predict.py',
            'Cài phụ thuộc PhoBERT/Java/VnCoreNLP, kiểm tra tài nguyên; fine-tune cùng split. '
            'Chọn checkpoint bằng validation, ghi truncation, tích hợp predict và tổng hợp '
            'kết quả cùng người 3. Pretrained/smoke chưa phải fine-tune đầy đủ.'),
    }
    for number, (title, files, work) in tasks.items():
        (guides / f'THANH_VIEN_{number}.md').write_text(
            f'# Thành viên {number} — {title}\n\n'
            'Clone cùng kho code, nhận cùng data/, cài package theo README code. '
            f'Làm nhánh member-{number}; sửa code gốc, không phát triển trên bản snapshot bàn giao.\n\n'
            f'File phụ trách: `{files}`.\n\n{work}\n\n'
            f'Cùng rà nhãn độc lập trong `data/labeled/annotation_batches_ai_v1/annotator_{number}.csv`. '
            'Đọc hướng dẫn nhãn, giữ ID/post_id/text/role; chỉ điền human_confirmed và reviewer '
            'sau khi chính mình đã đọc/xác nhận. Câu không rõ giữ pending/label trống. '
            'Gửi file riêng lại người 1 qua kênh nội bộ. Không dùng confirmed_v1 cũ.\n\n'
            'Dùng bộ chia bất biến 1.801/409/390; chưa có nhãn người xác nhận ban đầu. '
            'Không commit dữ liệu thật/model/output lên kho public. '
            'Đầu ra code gửi PR; nhật ký/cấu hình và phần báo cáo ghi rõ việc đã chạy, chưa chạy.\n',
            encoding='utf-8')
    (guides / 'GHEP_NHAN_NGUOI.md').write_text(
        '# Người 1 ghép đợt rà độc lập\n\nSau khi nhận file của cả bốn người, '
        'đặt vào data/labeled/annotation_batches_ai_v1 trên clone có gói data. '
        'Chạy từ gốc code:\n\n```powershell\n'
        '.\\.venv\\Scripts\\python.exe -m ptit_sentiment.data.annotation_review merge '
        '--assignment-dir data/labeled/annotation_batches_ai_v1 '
        '--output-dir data/labeled/human_review_handoff_v1\n```\n\n'
        'Output mới ghi needs_review/disagreements; pending chưa thành nhãn người. '
        'Thảo luận bất đồng, chỉ xác nhận thật rồi ghép theo docs/annotation_review.md. '
        'Không sửa tập AI hiện tại; phát hành bộ nhãn/bộ chia version mới. '
        'Bản sao assignment_manifest đã đổi clean_file sang đường dẫn tương đối để chạy trên máy khác.\n',
        encoding='utf-8')
    (destination / 'README_BAN_GIAO.md').write_text(
        '# Bàn giao PTIT Sentiment Analysis\n\n'
        '**Chỉ đưa nội dung 01_CODE_GITHUB lên gốc kho GitHub.** '
        'Không upload toàn bộ thư mục bàn giao.\n\n'
        '| Phần | Mục đích |\n|---|---|\n'
        '| 01_CODE_GITHUB | Code/config/tests/docs/ví dụ tổng hợp để commit |\n'
        '| 02_DATA_PRIVATE | Nhãn AI, train/validation/test, metadata, đợt rà nhãn người; chia sẻ riêng |\n'
        '| 03_HUONG_DAN_THANH_VIEN | Việc và đầu ra cho từng người, hướng dẫn merge |\n\n'
        'Code là snapshot có chọn lọc từ workspace, không phải dự án mới để phát triển song song. '
        'Các thành viên tiếp tục cùng Git repository. Bộ `ai_initial_4e93595ef7d1_seed42`: '
        '2.600 nhãn AI (362 positive/1.800 neutral/438 negative), '
        'train 1.801/validation 409/test 390, 194 unresolved và 70 excluded. '
        'Chưa có nhãn người xác nhận; test AI chưa thay đánh giá nhãn người độc lập.\n\n'
        'Sao chép data/ từ 02_DATA_PRIVATE vào gốc clone code trên từng máy. '
        'Xem README code cho lệnh cài/validate/train/evaluate/predict. '
        'Dữ liệu có thông tin nhận diện tiềm năng, chỉ chia sẻ nội bộ được phép; '
        'raw/ảnh/cookie/cache/model/báo cáo sai và script giả xác nhận không được đóng gói.\n\n'
        'MANIFEST_BAN_GIAO.json liệt kê file, SHA-256 và phiên bản. '
        'Chưa commit/push/upload; chưa thu thập lại, fit hoặc train khi đóng gói.\n',
        encoding='utf-8')
    inventory = {p.relative_to(destination).as_posix(): file_hash(p)
                 for p in sorted(destination.rglob('*')) if p.is_file()}
    write_json(destination / 'MANIFEST_BAN_GIAO.json', {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset_version': manifest['dataset_version'], 'split_manifest_sha256': digest,
        'policy': {'01_CODE_GITHUB': 'public_code', '02_DATA_PRIVATE': 'internal_only',
                   '03_HUONG_DAN_THANH_VIEN': 'team_instructions'},
        'source_files_unchanged': True, 'collection_or_training_performed': False,
        'files_sha256': inventory,
    })
    return destination, len(inventory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--output', default='handoff/PTIT_BAN_GIAO_20261007')
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = Path(args.root) / output
    destination, count = prepare(args.root, output)
    print(json.dumps({'folder': str(destination), 'files': count,
                      'note': 'Chỉ 01_CODE_GITHUB để upload; dữ liệu chia sẻ nội bộ.'}, ensure_ascii=False))


if __name__ == '__main__':
    run_cli(main)
