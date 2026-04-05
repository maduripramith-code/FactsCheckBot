from flask import Flask, request, render_template, jsonify
import joblib
import re

app = Flask(__name__)

# LOAD MODEL
data = joblib.load("model.pkl")
model = data["model"]
vectorizer = data["vectorizer"]

# TEXT CLEANING
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\n', '', text)
    return text

# KNOWLEDGE BASE
FACT_DB = {
    "prime minister of india": "Narendra Modi is the Prime Minister of India.",
    "president of india": "Droupadi Murmu is the President of India.",
    "capital of india": "New Delhi is the capital of India.",
    "earth is flat": "The Earth is spherical, not flat.",
    "india independence": "India gained independence in 1947.",
    "sun rises in west": "The Sun rises in the east, not west."
}

# EXPLANATION FUNCTION
def explain_claim(text, prediction):
    text_lower = text.lower()

    for key, fact in FACT_DB.items():
        if key in text_lower:

            if "not" in text_lower or "no" in text_lower:
                if prediction == "FAKE":
                    return f"❌ This claim is incorrect because {fact}"
                else:
                    return f"⚠️ This claim may be misleading. {fact}"
            else:
                if prediction == "FACT":
                    return f"✅ This claim is correct. {fact}"
                else:
                    return f"⚠️ This claim may be incorrect. {fact}"

    if prediction == "FAKE":
        return "❌ This claim appears misleading or not supported by reliable information."
    else:
        return "✅ This claim appears consistent with known factual patterns."

# ROUTE 1 → INDEX PAGE
@app.route("/")
def index():
    return render_template("index.html")

# ROUTE 2 → MAIN PAGE
@app.route("/main")
def main():
    return render_template("main.html")

# ROUTE 3 → API (IMPORTANT)
@app.route("/check", methods=["POST"])
def check():
    user_input = request.form.get("news")

    if not user_input:
        return jsonify({
            "result": "ERROR",
            "explanation": "No input received"
        })

    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned])

    result = model.predict(vectorized)[0]
    prediction = "FAKE" if result == 0 else "FACT"

    explanation = explain_claim(user_input, prediction)

    return jsonify({
        "result": prediction,
        "explanation": explanation
    })

# RUN
if __name__ == "__main__":
    app.run(debug=True)