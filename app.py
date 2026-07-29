import hashlib
import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


LOGGER = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent

DEFAULT_MODEL_FILENAME = "intent_model_tfidf_logreg_training_13.joblib"
DEFAULT_METADATA_FILENAME = (
    "intent_model_tfidf_logreg_training_13.metadata.json"
)

EXPECTED_INTENTS = {
    "greeting",
    "product_discovery",
    "recommendation",
    "product_detail",
    "price_promo",
    "stock_availability",
    "shipping_transaction",
    "shipping_origin",
    "return_product",
    "compare",
    "transaction_status",
    "shipment_tracking",
    "general",
}


def resolve_artifact_path(env_name: str, default_filename: str) -> Path:
    configured = os.getenv(env_name, default_filename).strip()
    path = Path(configured)
    return path if path.is_absolute() else BASE_DIR / path


MODEL_PATH = resolve_artifact_path(
    "INTENT_MODEL_PATH",
    DEFAULT_MODEL_FILENAME,
)
METADATA_PATH = resolve_artifact_path(
    "INTENT_MODEL_METADATA_PATH",
    DEFAULT_METADATA_FILENAME,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_metadata(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"Metadata model tidak ditemukan: {path.name}")

    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Metadata model tidak valid: {path.name}") from exc

    if not isinstance(metadata, dict):
        raise RuntimeError("Metadata model harus berupa JSON object")

    return metadata


def resolve_confidence_threshold(metadata: dict) -> float:
    raw_value = os.getenv(
        "INTENT_CONFIDENCE_THRESHOLD",
        metadata.get("recommended_confidence_threshold", 0.6),
    )

    try:
        threshold = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "INTENT_CONFIDENCE_THRESHOLD harus berupa angka"
        ) from exc

    if not 0 < threshold < 1:
        raise RuntimeError(
            "INTENT_CONFIDENCE_THRESHOLD harus lebih dari 0 dan kurang dari 1"
        )

    return threshold


def load_and_validate_model(model_path: Path, metadata: dict):
    if not model_path.is_file():
        raise RuntimeError(f"Artifact model tidak ditemukan: {model_path.name}")

    expected_hash = str(metadata.get("model_sha256") or "").strip().lower()
    if not expected_hash:
        raise RuntimeError("Metadata model tidak memiliki model_sha256")
    if sha256_file(model_path) != expected_hash:
        raise RuntimeError("Checksum artifact model tidak sesuai metadata")

    metadata_labels = {str(label) for label in metadata.get("labels", [])}
    if metadata_labels != EXPECTED_INTENTS:
        raise RuntimeError("Daftar label pada metadata tidak sesuai kontrak intent")

    trained_sklearn = str(
        metadata.get("versions", {}).get("scikit_learn") or ""
    ).strip()
    if trained_sklearn and trained_sklearn != sklearn.__version__:
        raise RuntimeError(
            "Versi scikit-learn berbeda: "
            f"artifact={trained_sklearn}, runtime={sklearn.__version__}"
        )

    loaded_model = joblib.load(model_path)
    if not callable(getattr(loaded_model, "predict", None)):
        raise RuntimeError("Artifact model tidak memiliki method predict")
    if not callable(getattr(loaded_model, "predict_proba", None)):
        raise RuntimeError("Artifact model tidak memiliki method predict_proba")

    classes = {str(label) for label in getattr(loaded_model, "classes_", [])}
    if classes != EXPECTED_INTENTS:
        missing = sorted(EXPECTED_INTENTS - classes)
        unknown = sorted(classes - EXPECTED_INTENTS)
        raise RuntimeError(
            f"Kontrak intent model tidak sesuai; missing={missing}, unknown={unknown}"
        )

    probe = loaded_model.predict_proba(["cek stok chogokin"])[0]
    if len(probe) != len(EXPECTED_INTENTS):
        raise RuntimeError("Jumlah probabilitas model tidak sesuai kontrak intent")

    return loaded_model


metadata = load_metadata(METADATA_PATH)
confidence_threshold = resolve_confidence_threshold(metadata)
model = load_and_validate_model(MODEL_PATH, metadata)
model_classes = np.asarray(model.classes_, dtype=str)

app = FastAPI(
    title="Robot Jadul Intent ML API",
    version=str(metadata.get("artifact_version", 1)),
)


class PredictRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


@app.get("/")
def root():
    return {
        "message": "Intent ML API aktif",
        "status": "ok",
        "model": MODEL_PATH.name,
        "intent_count": len(model_classes),
        "confidence_threshold": confidence_threshold,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "model": MODEL_PATH.name,
        "metadata": METADATA_PATH.name,
        "intent_count": len(model_classes),
    }


@app.post("/predict_intent")
def predict_intent(payload: PredictRequest):
    question = normalize_text(payload.question)
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty")

    try:
        probabilities = model.predict_proba([question])[0]
        best_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[best_idx])
        predicted_intent = str(model_classes[best_idx])

        ranked_indices = np.argsort(probabilities)[::-1][:3]
        top3 = [
            {
                "intent": str(model_classes[index]),
                "prob": float(probabilities[index]),
            }
            for index in ranked_indices
        ]

        return {
            "intent": predicted_intent,
            "confidence": confidence,
            "top3": top3,
            "method": "tfidf_logreg",
            "is_low_confidence": confidence < confidence_threshold,
            "model_name": "TF-IDF Word/Character + Logistic Regression",
            "model_version": metadata.get("artifact_version", 1),
        }
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Intent inference failed")
        raise HTTPException(
            status_code=500,
            detail="Inference error",
        ) from exc
