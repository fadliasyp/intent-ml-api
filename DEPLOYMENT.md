# Deployment Intent ML API

## File Wajib

Pastikan file berikut ikut masuk ke deployment:

- `app.py`
- `requirements.txt`
- `intent_model_tfidf_logreg_training_13.joblib`
- `intent_model_tfidf_logreg_training_13.metadata.json`

Model lama boleh tetap disimpan sebagai arsip, tetapi tidak lagi dimuat oleh API.

## Start Command

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

Gunakan Python 3.12 dan dependency yang dipin di `requirements.txt`.

## Environment Variable

Semua environment variable berikut bersifat opsional:

```text
INTENT_MODEL_PATH=intent_model_tfidf_logreg_training_13.joblib
INTENT_MODEL_METADATA_PATH=intent_model_tfidf_logreg_training_13.metadata.json
INTENT_CONFIDENCE_THRESHOLD=0.60
```

Path relatif dihitung dari folder `app.py`. API akan gagal saat startup jika
artifact hilang, checksum berbeda, versi scikit-learn tidak cocok, atau model
tidak memiliki tepat 13 intent.

## Health Check

Set health-check deployment ke:

```text
GET /health
```

Respons sehat:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model": "intent_model_tfidf_logreg_training_13.joblib",
  "metadata": "intent_model_tfidf_logreg_training_13.metadata.json",
  "intent_count": 13
}
```

Endpoint yang dipakai chatbot tetap:

```text
POST /predict_intent
```

Body:

```json
{
  "question": "metode pembayarannya apa aja"
}
```
