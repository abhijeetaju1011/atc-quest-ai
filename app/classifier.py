import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import os

# Path to dataset
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "questions.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "classifier.pkl")
SPAM_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "spam_model.pkl")

def train_models():
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded: {len(df)} rows")

    X = df["text"]

    # ── Model 1: Question Category Classifier ──
    y_category = df["category"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_category, test_size=0.2, random_state=42
    )

    category_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", MultinomialNB())
    ])
    category_pipeline.fit(X_train, y_train)
    y_pred = category_pipeline.predict(X_test)
    print("\n── Category Classifier Results ──")
    print(classification_report(y_test, y_pred))

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(category_pipeline, f)
    print(f"Category model saved!")

    # ── Model 2: Spam Classifier ──
    y_spam = df["is_spam"]
    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X, y_spam, test_size=0.2, random_state=42
    )

    spam_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", MultinomialNB())
    ])
    spam_pipeline.fit(X_train2, y_train2)
    y_pred2 = spam_pipeline.predict(X_test2)
    print("\n── Spam Classifier Results ──")
    print(classification_report(y_test2, y_pred2))

    # Save model
    with open(SPAM_MODEL_PATH, "wb") as f:
        pickle.dump(spam_pipeline, f)
    print(f"Spam model saved!")

    return category_pipeline, spam_pipeline


def load_models():
    with open(MODEL_PATH, "rb") as f:
        category_model = pickle.load(f)
    with open(SPAM_MODEL_PATH, "rb") as f:
        spam_model = pickle.load(f)
    return category_model, spam_model


def classify_question(text: str):
    category_model, spam_model = load_models()

    # Get category prediction + confidence
    category = category_model.predict([text])[0]
    category_proba = max(category_model.predict_proba([text])[0])

    # Get spam prediction + confidence
    is_spam = spam_model.predict([text])[0]
    spam_proba = max(spam_model.predict_proba([text])[0])

    # If confidence is low — flag for human review
    needs_review = category_proba < 0.6 or spam_proba < 0.6

    return {
        "text": text,
        "category": str(category),
        "category_confidence": round(float(category_proba), 2),
        "is_spam": bool(is_spam),
        "spam_confidence": round(float(spam_proba), 2),
        "needs_human_review": bool(needs_review)
    }


if __name__ == "__main__":
    train_models()