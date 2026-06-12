import pandas as pd
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "quiz_data.csv")

# Study recommendations for each topic
RECOMMENDATIONS = {
    "Python Basics": "Revise variables, data types, and basic syntax.",
    "Loops": "Practice for/while loops with exercises on LeetCode.",
    "Functions": "Study function arguments, return values, and recursion.",
    "OOP": "Learn classes, objects, inheritance, and polymorphism.",
    "Machine Learning": "Start with scikit-learn tutorials and linear regression.",
    "Data Structures": "Practice arrays, linked lists, stacks, and queues.",
    "Databases": "Learn SQL basics — SELECT, JOIN, GROUP BY queries."
}

def load_data():
    df = pd.read_csv(DATA_PATH)
    df["mastery"] = round((df["score"] / df["total"]) * 100, 1)
    return df

def get_weak_topics(student_id: str):
    df = load_data()

    # Filter student
    student_df = df[df["student_id"] == student_id]

    if student_df.empty:
        return {"error": f"Student {student_id} not found!"}

    student_name = student_df["student_name"].iloc[0]

    # Sort by mastery (lowest first = weakest)
    weak = student_df.sort_values("mastery")

    weak_topics = []
    for _, row in weak.iterrows():
        weak_topics.append({
            "topic": row["topic"],
            "score": int(row["score"]),
            "total": int(row["total"]),
            "mastery_percent": float(row["mastery"]),
            "attempts": int(row["attempts"])
        })

    return {
        "student_id": student_id,
        "student_name": student_name,
        "weak_topics": weak_topics
    }

def get_recommendations(student_id: str):
    df = load_data()

    student_df = df[df["student_id"] == student_id]

    if student_df.empty:
        return {"error": f"Student {student_id} not found!"}

    student_name = student_df["student_name"].iloc[0]

    # Topics with mastery below 60% need improvement
    weak_df = student_df[student_df["mastery"] < 60].sort_values("mastery")

    recommendations = []
    for _, row in weak_df.iterrows():
        topic = row["topic"]
        recommendations.append({
            "topic": topic,
            "mastery_percent": float(row["mastery"]),
            "priority": "High" if row["mastery"] < 40 else "Medium",
            "recommendation": RECOMMENDATIONS.get(topic, "Practice more exercises."),
            "why": f"You scored {int(row['score'])}/{int(row['total'])} — needs improvement."
        })

    return {
        "student_id": student_id,
        "student_name": student_name,
        "total_weak_topics": len(recommendations),
        "recommendations": recommendations
    }

if __name__ == "__main__":
    print("── Weak Topics for S001 ──")
    print(get_weak_topics("S001"))
    print("\n── Recommendations for S001 ──")
    print(get_recommendations("S001")) 
