
from pathlib import Path

from ptit_sentiment.models.phobert import PhoBERTPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[3]

ARTIFACT_DIR = (
    PROJECT_ROOT / "outputs" / "phobert" / "best_model"
)

SEGMENTER_DIR = (
    PROJECT_ROOT / "artifacts" / "vncorenlp"
)


def main():

    print("Project root:", PROJECT_ROOT)
    print("Artifact:", ARTIFACT_DIR)
    print("Artifact exists:", ARTIFACT_DIR.exists())

    predictor = PhoBERTPredictor(
        artifact_dir=ARTIFACT_DIR,
        segmenter_dir=SEGMENTER_DIR,
    )

    print("\nPhoBERT is ready!")
    print("Nhập 'exit' để thoát.")

    while True:

        text = input("\nNhập bình luận: ").strip()

        if text.lower() == "exit":
            break

        if not text:
            print("Vui lòng nhập bình luận.")
            continue

        labels, probabilities, label_names = predictor.predict(
            [text]
        )

        print("\n===== KẾT QUẢ =====")

        print("Model: PhoBERT")
        print("Sentiment:", labels[0])

        print("\nProbabilities:")

        for label, probability in zip(
            label_names,
            probabilities[0],
        ):
            print(f"{label}: {probability:.2%}")

        print(
            "\nConfidence:",
            f"{probabilities[0].max():.2%}",
        )


if __name__ == "__main__":
    main()
