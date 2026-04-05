# FactsCheckBot
FactsCheckBot is an intelligent system designed to automatically verify the authenticity of information and classify it as fact or fake. It leverages techniques from Natural Language Processing and Machine Learning to analyze textual data and detect misleading or false claims.

# Project Overview

In today’s digital world, misinformation spreads rapidly across social media and news platforms. FactsCheckBot aims to reduce this problem by automatically analyzing user-input text and classifying it as Fact or Fake, along with a short explanation.

# Features
 Real-time fact-checking of user input text
 Machine Learning-based classification model
 Explanation for prediction results
 Simple and interactive web interface (Flask-based)
 Fast response using trained vectorizer + model pipeline
 
# Technologies Used
Python 
Flask (Backend Framework)
Scikit-learn (ML Model)
Pandas & NumPy (Data Processing)
HTML, CSS (Frontend)
Joblib (Model Saving/Loading)

# How It Works
User enters a statement in the web interface
Text is cleaned and preprocessed
Vectorizer converts text into numerical features
ML model predicts whether it is Fact or Fake
Result is displayed with a short explanation

# How to Run the Project
# Clone the repository
git clone https://github.com/your-username/FactsCheckBot.git

# Navigate into folder
cd FactsCheckBot

# Install dependencies
pip install -r requirements.txt

# Run Flask app
python app.py
