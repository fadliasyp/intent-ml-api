# train_model.py
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# baca dataset
df = pd.read_csv("dataset_intent_1800_realistic_robotjadul.csv")

X = df["text"].astype(str)
y = df["label"].astype(str)

# bagi train dan test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# buat pipeline TF-IDF + Logistic Regression
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2)
    )),
    ("clf", LogisticRegression(
        max_iter=500
    ))
])

# latih model
model.fit(X_train, y_train)

# evaluasi
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred, digits=3))

# simpan model
joblib.dump(model, "intent_tfidf_logreg.joblib")
print("Model tersimpan: intent_tfidf_logreg.joblib")