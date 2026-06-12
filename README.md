# 🎓 ATC QUEST AI Learning Assistant

> Smart AI layer for the ATC QUEST learning platform  
> **Internship Project — Abhijeet Anand | ATCUALITY | Summer 2026**

---

## 📌 What This Project Does

The ATC QUEST AI Learning Assistant is a smart backend system that does three things:

1. **Sorts student questions** — doubt, assignment, or admin
2. **Filters spam & toxic content** — keeps the platform clean
3. **Finds weak topics & recommends what to study** — helps students learn better

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.12 |
| API Framework | FastAPI + Uvicorn |
| ML / NLP | scikit-learn, TF-IDF, Naive Bayes |
| Data | CSV (simulated dataset) |
| Dashboard | HTML + Chart.js |
| Tests | pytest |

---

## 📁 Project Structure

```
atc-quest-ai/
├── app/
│   ├── classifier.py      # Question classifier + spam detector
│   ├── moderator.py       # Toxic content filter
│   ├── insights.py        # Weak topics + recommendations
│   ├── main.py            # FastAPI app + all endpoints
│   ├── data/
│   │   ├── questions.csv  # Labelled question dataset
│   │   └── quiz_data.csv  # Student quiz results
│   ├── models/            # Trained ML models (.pkl)
│   └── templates/
│       └── dashboard.html # Web dashboard
├── tests/
│   └── test_api.py
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/abhijeetaju1011/atc-quest-ai.git
cd atc-quest-ai
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the models
```bash
python app/classifier.py
python app/moderator.py
```

### 5. Start the server
```bash
uvicorn app.main:app --reload
```

### 6. Open in browser
- **Dashboard** → http://127.0.0.1:8000
- **API Docs** → http://127.0.0.1:8000/docs

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Dashboard |
| GET | `/health` | Health check |
| POST | `/classify-question` | Classify a question |
| POST | `/moderate` | Check spam/toxic |
| GET | `/weak-topics/{student_id}` | Get weak topics |
| GET | `/recommendations/{student_id}` | Get study recommendations |

---

## 📊 Model Results

| Model | Accuracy |
|-------|----------|
| Question Classifier | ~70% |
| Spam Detector | ~70% |
| Toxic Content Filter | ~80% |

> Note: Models trained on small simulated dataset. Accuracy improves with more data.

---

## 👤 Author

**Abhijeet Anand**  
B.Tech CSE — Arka Jain University, Jamshedpur  
Software Engineering Intern (AI/NLP) — ATCUALITY  
📧 abhijeetanand1011@gmail.com

---

*Prepared as part of ATCUALITY Summer Internship 2026*
