from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from classifier import classify_question, train_models
from moderator import moderate_text, train_toxic_model
from insights import get_weak_topics, get_recommendations

app = FastAPI(
    title="ATC QUEST AI Learning Assistant",
    description="Smart layer for ATC QUEST learning platform",
    version="1.0.0"
)

class TextInput(BaseModel):
    text: str

@app.on_event("startup")
def startup_event():
    model_path = os.path.join("app", "models", "classifier.pkl")
    toxic_path = os.path.join("app", "models", "toxic_model.pkl")
    if not os.path.exists(model_path):
        train_models()
    if not os.path.exists(toxic_path):
        train_toxic_model()
    print("All models ready!")

@app.get("/", response_class=HTMLResponse)
def home():
    template_path = os.path.join(os.path.dirname(__file__), "templates", "dashboard.html")
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "1.0.0"}

@app.post("/classify-question")
def classify(input: TextInput):
    return classify_question(input.text)

@app.post("/moderate")
def moderate(input: TextInput):
    return moderate_text(input.text)

@app.get("/weak-topics/{student_id}")
def weak_topics(student_id: str):
    return get_weak_topics(student_id)

@app.get("/recommendations/{student_id}")
def recommendations(student_id: str):
    return get_recommendations(student_id)