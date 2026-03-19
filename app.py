# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# load model sekali saat app start
model = joblib.load("intent_tfidf_logreg_final.joblib")

class PredictRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "Intent ML API aktif"}

@app.post("/predict_intent")
def predict_intent(payload: PredictRequest):
    question = payload.question.strip()

    pred = model.predict([question])[0]
    probs = model.predict_proba([question])[0]

    classes = model.named_steps["clf"].classes_
    best_idx = int(np.argmax(probs))
    confidence = float(probs[best_idx])

    top3 = sorted(
        [{"intent": str(c), "prob": float(p)} for c, p in zip(classes, probs)],
        key=lambda x: x["prob"],
        reverse=True
    )[:3]

    return {
        "intent": str(pred),
        "confidence": confidence,
        "top3": top3,
        "method": "tfidf_logreg"
    }