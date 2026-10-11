"""Pipeline trọn vẹn và baseline most_frequent."""
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from ptit_sentiment.features.vectorizers import make_vectorizer, make_tfidf_word_char
from ptit_sentiment.preprocessing.tokenize import TextPreprocessor

MODEL_NAMES = ("bow_nb", "tfidf_nb", "bow_svm", "tfidf_svm")


def make_pipeline(feature, algorithm, config, ngram_range, parameter, *, emoji=None, class_weight=None,
                  tfidf_mode="word", vectorizer_options=None):
    """Tạo pipeline mới cho từng thử nghiệm, chưa fit."""
    if algorithm == "nb":
        classifier = MultinomialNB(alpha=parameter)
    elif algorithm == "svm":
        classifier = LinearSVC(C=parameter, class_weight=class_weight, random_state=config.get("seed", 42),
                               dual="auto", max_iter=10000)
    else:
        raise ValueError(f"Thuật toán không hỗ trợ: {algorithm}.")
    preprocessing = dict(config.get("preprocessing", {}))
    if emoji is not None:
        preprocessing["emoji"] = emoji
    options = dict(config.get("vectorizer", {})) if vectorizer_options is None else dict(vectorizer_options)
    if feature == "tfidf" and tfidf_mode == "word_char":
        vectorizer = make_tfidf_word_char(ngram_range, **options)
    elif tfidf_mode == "word":
        vectorizer = make_vectorizer(feature, ngram_range, **options)
    else:
        raise ValueError(f"tfidf_mode không hỗ trợ: {tfidf_mode}.")
    return Pipeline([
        ("preprocess", TextPreprocessor(**preprocessing)),
        ("vectorizer", vectorizer),
        ("classifier", classifier),
    ])


def fit_baseline(texts, labels):
    """Học nhãn phổ biến nhất chỉ từ train; input text bị DummyClassifier bỏ qua."""
    return DummyClassifier(strategy="most_frequent").fit([[text] for text in texts], labels)


def predict_classical(model, texts, baseline=False):
    """Trả nhãn và xác suất nếu classifier thực sự hỗ trợ; SVM trả None."""
    inputs = [[text] for text in texts] if baseline else list(texts)
    predictions = model.predict(inputs)
    probabilities = model.predict_proba(inputs) if hasattr(model, "predict_proba") else None
    return predictions, probabilities, model.classes_
