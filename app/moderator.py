from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import pickle
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "questions.csv")
TOXIC_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "toxic_model.pkl")

def train_toxic_model():
    df = pd.read_csv(DATA_PATH)

    X = df["text"]
    y = df["is_toxic"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", MultinomialNB())
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("\n── Toxic Content Classifier Results ──")
    print(classification_report(y_test, y_pred, zero_division=0))

    with open(TOXIC_MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print("Toxic model saved!")

    return pipeline


def moderate_text(text: str):
    with open(TOXIC_MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    is_toxic = model.predict([text])[0]
    confidence = max(model.predict_proba([text])[0])
    needs_review = confidence < 0.65

    return {
        "text": text,
        "is_toxic": bool(is_toxic),
        "confidence": round(float(confidence), 2),
        "action": "block" if is_toxic else "allow",
        "needs_human_review": needs_review
    }


if __name__ == "__main__":
    train_toxic_model()
