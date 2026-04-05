import pandas as pd
import numpy as np
import re
import string
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# LOAD DATA
true = pd.read_csv("true.csv")
fake = pd.read_csv("fake.csv")

# LABELING
true['label'] = 1
fake['label'] = 0

# MERGE DATA
df = pd.concat([fake, true], axis=0)

# DROP UNUSED COLUMNS
df = df.drop(["title", "subject", "date"], axis=1)

# SHUFFLE DATA
df = df.sample(frac=1).reset_index(drop=True)

# CLEAN TEXT FUNCTION
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# APPLY CLEANING
df["text"] = df["text"].apply(wordopt)

# SPLIT DATA
x = df["text"]
y = df["label"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# VECTORIZATION
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# LOGISTIC REGRESSION
LR = LogisticRegression()
LR.fit(xv_train, y_train)

pred_lr = LR.predict(xv_test)
print("Logistic Regression Results:")
print(classification_report(y_test, pred_lr))

# DECISION TREE
DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)

pred_dt = DT.predict(xv_test)
print("Decision Tree Results:")
print(classification_report(y_test, pred_dt))

# SAVE BEST MODEL (LR)
joblib.dump({
    "model": LR,
    "vectorizer": vectorization
}, "model.pkl")

print("✅ model.pkl created successfully!")
