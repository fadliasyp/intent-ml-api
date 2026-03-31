#app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("intent_tfidf_logreg_final.joblib")

class PredictRequest(BaseModel):
    question: str

def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())

@app.get("/")
def root():
    return {"message": "Intent ML API aktif"}

@app.post("/predict_intent")
def predict_intent(payload: PredictRequest):
    try:
        question = normalize_text(payload.question)

        if not question:
            raise HTTPException(status_code=400, detail="Question is empty")

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

        threshold = 0.6

        return {
            "intent": str(pred),
            "confidence": confidence,
            "top3": top3,
            "method": "tfidf_logreg",
            "is_low_confidence": confidence < threshold,
            "model_name": "TF-IDF + Logistic Regression"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    
    # Model hasil training disimpan dalam format .joblib dan diintegrasikan ke dalam layanan API berbasis FastAPI. 
    # Model dimuat satu kali saat aplikasi berjalan, sehingga proses prediksi intent dapat dilakukan secara real-time dengan lebih efisien.
    #  API mengembalikan label intent, nilai confidence, serta tiga prediksi teratas untuk mendukung proses analisis dan fallback pada sistem chatbot.