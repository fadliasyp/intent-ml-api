"""Generate the reproducible 13-intent training notebook.

The notebook is generated instead of edited by hand so its structure stays
reviewable and deterministic in git.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "Training_Intent_13.ipynb"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


cells = [
    markdown(
        """
        # Training Model Intent Chatbot - 13 Intent

        Notebook ini melatih classifier intent teks tanpa mengubah logic chatbot.
        Dataset training dan **hard test** dipisahkan. Pemilihan model mengutamakan
        Macro F1 pada hard test agar intent dengan jumlah data lebih sedikit tetap
        diperhitungkan.

        Output production:

        - `intent_model_tfidf_logreg_training_13.joblib`
        - `intent_model_tfidf_logreg_training_13.metadata.json`
        """
    ),
    markdown(
        """
        ## 1. Siapkan environment

        Jalankan cell ini sebelum mengimpor scikit-learn. Versi yang dipin harus
        sama dengan dependency API saat artifact dipasang pada tahap deployment.
        """
    ),
    code(
        """
        %pip install -q scikit-learn==1.8.0 pandas==3.0.1 joblib==1.5.3 numpy==2.4.2 matplotlib seaborn
        """
    ),
    code(
        """
        import hashlib
        import json
        import platform
        from datetime import datetime, timezone
        from pathlib import Path

        import joblib
        import numpy as np
        import pandas as pd
        import sklearn
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
        )
        from sklearn.model_selection import StratifiedKFold, cross_validate
        from sklearn.pipeline import FeatureUnion, Pipeline

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            plt = None
            sns = None

        RANDOM_STATE = 42
        np.random.seed(RANDOM_STATE)

        print("Python:", platform.python_version())
        print("scikit-learn:", sklearn.__version__)
        print("pandas:", pd.__version__)
        print("joblib:", joblib.__version__)
        print("numpy:", np.__version__)
        """
    ),
    markdown(
        """
        ## 2. Muat dataset canonical

        Jika notebook dibuka dari folder repository, file akan ditemukan otomatis.
        Di Google Colab, upload kedua CSV ketika dialog file muncul.
        """
    ),
    code(
        """
        TRAIN_FILENAME = "dataset_intent_13_ready_training.csv"
        HARD_TEST_FILENAME = "dataset_intent_13_hard_test.csv"

        def find_dataset(filename):
            candidates = [
                Path("Data yg dilatih dan di test") / filename,
                Path(filename),
                Path("/content") / filename,
            ]
            return next((path for path in candidates if path.exists()), None)

        train_path = find_dataset(TRAIN_FILENAME)
        hard_test_path = find_dataset(HARD_TEST_FILENAME)

        if train_path is None or hard_test_path is None:
            try:
                from google.colab import files
            except ImportError as exc:
                raise FileNotFoundError(
                    "Dataset tidak ditemukan. Jalankan notebook dari root repository "
                    "atau letakkan kedua CSV di working directory."
                ) from exc

            print("Upload dataset training dan hard test:")
            uploaded = files.upload()
            for filename, content in uploaded.items():
                Path(filename).write_bytes(content)

            train_path = find_dataset(TRAIN_FILENAME)
            hard_test_path = find_dataset(HARD_TEST_FILENAME)

        if train_path is None or hard_test_path is None:
            raise FileNotFoundError(
                f"Wajib tersedia: {TRAIN_FILENAME} dan {HARD_TEST_FILENAME}"
            )

        train_df = pd.read_csv(train_path)
        hard_test_df = pd.read_csv(hard_test_path)
        print("Training:", train_path, train_df.shape)
        print("Hard test:", hard_test_path, hard_test_df.shape)
        """
    ),
    markdown(
        """
        ## 3. Validasi kontrak 13 intent

        Cell ini sengaja berhenti dengan error jika ada label asing, teks kosong,
        label konflik, kelas terlalu kecil, atau kebocoran teks dari training ke
        hard test.
        """
    ),
    code(
        """
        INTENT_LABELS = [
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
        ]

        def normalize_text(value):
            return " ".join(str(value).lower().split())

        for frame_name, frame in [
            ("training", train_df),
            ("hard test", hard_test_df),
        ]:
            assert list(frame.columns) == ["text", "label"], (
                f"Kolom {frame_name} harus persis ['text', 'label']"
            )
            assert frame["text"].notna().all(), f"Ada teks kosong di {frame_name}"
            assert frame["label"].notna().all(), f"Ada label kosong di {frame_name}"
            assert (frame["text"].astype(str).str.strip() != "").all()
            unknown = sorted(set(frame["label"]) - set(INTENT_LABELS))
            assert not unknown, f"Label asing di {frame_name}: {unknown}"
            missing = sorted(set(INTENT_LABELS) - set(frame["label"]))
            assert not missing, f"Label hilang di {frame_name}: {missing}"

        normalized_train = train_df["text"].map(normalize_text)
        normalized_hard = hard_test_df["text"].map(normalize_text)

        conflict_count = (
            train_df.assign(normalized_text=normalized_train)
            .groupby("normalized_text")["label"]
            .nunique()
            .gt(1)
            .sum()
        )
        overlap = set(normalized_train) & set(normalized_hard)
        min_class_size = int(train_df["label"].value_counts().min())

        assert conflict_count == 0, f"Ada {conflict_count} teks training dengan label konflik"
        assert not overlap, f"Ada {len(overlap)} teks bocor dari training ke hard test"
        assert min_class_size >= 100, f"Kelas terkecil hanya berisi {min_class_size} data"

        print("Validasi berhasil")
        print("Jumlah training:", len(train_df))
        print("Jumlah hard test:", len(hard_test_df))
        print("Exact overlap:", len(overlap))
        display(
            train_df["label"]
            .value_counts()
            .reindex(INTENT_LABELS)
            .rename("training_count")
            .to_frame()
        )
        """
    ),
    markdown(
        """
        ## 4. Kandidat model

        - **word_tfidf**: baseline ringan.
        - **word_char_tfidf**: menambahkan pola karakter agar lebih tahan typo,
          singkatan, dan bahasa chat.

        Cross-validation membantu melihat kestabilan internal. Karena sebagian data
        merupakan variasi kalimat, nilai hard test tetap menjadi dasar utama
        pemilihan model.
        """
    ),
    code(
        """
        def make_word_model():
            return Pipeline([
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 2),
                        min_df=1,
                        max_df=0.98,
                        sublinear_tf=True,
                        max_features=30_000,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        solver="lbfgs",
                        class_weight="balanced",
                        C=4.0,
                        max_iter=3_000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ])

        def make_word_char_model():
            features = FeatureUnion([
                (
                    "word",
                    TfidfVectorizer(
                        lowercase=True,
                        analyzer="word",
                        ngram_range=(1, 2),
                        min_df=1,
                        max_df=0.98,
                        sublinear_tf=True,
                        max_features=30_000,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        lowercase=True,
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=2,
                        sublinear_tf=True,
                        max_features=50_000,
                    ),
                ),
            ])
            return Pipeline([
                ("features", features),
                (
                    "classifier",
                    LogisticRegression(
                        solver="lbfgs",
                        class_weight="balanced",
                        C=4.0,
                        max_iter=3_000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ])

        candidates = {
            "word_tfidf": make_word_model(),
            "word_char_tfidf": make_word_char_model(),
        }

        X_train = train_df["text"].astype(str)
        y_train = train_df["label"].astype(str)
        X_hard = hard_test_df["text"].astype(str)
        y_hard = hard_test_df["label"].astype(str)
        """
    ),
    code(
        """
        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        cv_rows = []

        for name, model in candidates.items():
            scores = cross_validate(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
                n_jobs=-1,
            )
            cv_rows.append({
                "model": name,
                "cv_accuracy_mean": scores["test_accuracy"].mean(),
                "cv_accuracy_std": scores["test_accuracy"].std(),
                "cv_macro_f1_mean": scores["test_macro_f1"].mean(),
                "cv_macro_f1_std": scores["test_macro_f1"].std(),
            })

        cv_results = pd.DataFrame(cv_rows).sort_values(
            "cv_macro_f1_mean",
            ascending=False,
        )
        display(cv_results)
        """
    ),
    markdown(
        """
        ## 5. Evaluasi hard test dan pilih model

        Hard test tidak pernah dipakai untuk fitting. Urutan pemilihan:
        `Macro F1`, lalu `accuracy`, lalu rata-rata `Macro F1` cross-validation.
        """
    ),
    code(
        """
        hard_rows = []
        fitted_candidates = {}
        prediction_by_model = {}

        for name, model in candidates.items():
            model.fit(X_train, y_train)
            prediction = model.predict(X_hard)
            fitted_candidates[name] = model
            prediction_by_model[name] = prediction
            hard_rows.append({
                "model": name,
                "hard_accuracy": accuracy_score(y_hard, prediction),
                "hard_macro_f1": f1_score(
                    y_hard,
                    prediction,
                    average="macro",
                    labels=INTENT_LABELS,
                ),
            })

        hard_results = pd.DataFrame(hard_rows)
        model_results = hard_results.merge(cv_results, on="model").sort_values(
            ["hard_macro_f1", "hard_accuracy", "cv_macro_f1_mean"],
            ascending=False,
        )
        selected_name = model_results.iloc[0]["model"]
        selected_model = fitted_candidates[selected_name]
        selected_prediction = prediction_by_model[selected_name]

        print("Model terpilih:", selected_name)
        display(model_results)
        print(
            classification_report(
                y_hard,
                selected_prediction,
                labels=INTENT_LABELS,
                zero_division=0,
            )
        )
        """
    ),
    code(
        """
        matrix = confusion_matrix(
            y_hard,
            selected_prediction,
            labels=INTENT_LABELS,
        )
        if plt is not None and sns is not None:
            plt.figure(figsize=(13, 10))
            sns.heatmap(
                matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=INTENT_LABELS,
                yticklabels=INTENT_LABELS,
            )
            plt.title(f"Confusion Matrix Hard Test - {selected_name}")
            plt.xlabel("Prediksi")
            plt.ylabel("Label aktual")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.show()
        else:
            display(
                pd.DataFrame(
                    matrix,
                    index=INTENT_LABELS,
                    columns=INTENT_LABELS,
                )
            )
        """
    ),
    markdown(
        """
        ## 6. Audit error dan confidence

        API memakai threshold `0.60`. Prediksi di bawah threshold akan jatuh ke
        rule-based intent detector, sehingga model tidak dipaksa menjawab ketika
        ragu.
        """
    ),
    code(
        """
        probabilities = selected_model.predict_proba(X_hard)
        sorted_probabilities = np.sort(probabilities, axis=1)
        confidence = sorted_probabilities[:, -1]
        margin = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]

        audit_df = hard_test_df.copy()
        audit_df["prediction"] = selected_prediction
        audit_df["confidence"] = confidence
        audit_df["top1_top2_margin"] = margin
        audit_df["correct"] = audit_df["label"] == audit_df["prediction"]

        errors = audit_df.loc[
            ~audit_df["correct"],
            ["text", "label", "prediction", "confidence", "top1_top2_margin"],
        ].sort_values("confidence", ascending=False)

        print("Jumlah salah:", len(errors))
        display(errors)

        threshold_rows = []
        for threshold in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
            accepted = audit_df["confidence"] >= threshold
            threshold_rows.append({
                "threshold": threshold,
                "coverage": accepted.mean(),
                "accepted_count": int(accepted.sum()),
                "accepted_accuracy": (
                    audit_df.loc[accepted, "correct"].mean()
                    if accepted.any()
                    else np.nan
                ),
            })

        threshold_results = pd.DataFrame(threshold_rows)
        display(threshold_results)
        """
    ),
    markdown(
        """
        ## 7. Ekspor artifact production

        Model dipasang ke API pada tahap terpisah setelah hasil evaluasi ditinjau.
        Hard test tetap tidak digabung ke training agar dapat dipakai kembali untuk
        regression test model berikutnya.
        """
    ),
    code(
        """
        MODEL_FILENAME = "intent_model_tfidf_logreg_training_13.joblib"
        METADATA_FILENAME = "intent_model_tfidf_logreg_training_13.metadata.json"
        RECOMMENDED_THRESHOLD = 0.60

        # Fit ulang secara eksplisit hanya pada seluruh dataset training.
        final_model = (
            make_word_char_model()
            if selected_name == "word_char_tfidf"
            else make_word_model()
        )
        final_model.fit(X_train, y_train)

        def sha256_file(path):
            digest = hashlib.sha256()
            with open(path, "rb") as file_handle:
                for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        joblib.dump(final_model, MODEL_FILENAME)

        selected_metrics = model_results.iloc[0].to_dict()
        metadata = {
            "artifact_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_filename": MODEL_FILENAME,
            "selected_pipeline": selected_name,
            "labels": INTENT_LABELS,
            "recommended_confidence_threshold": RECOMMENDED_THRESHOLD,
            "training_rows": int(len(train_df)),
            "hard_test_rows": int(len(hard_test_df)),
            "training_dataset_sha256": sha256_file(train_path),
            "hard_test_dataset_sha256": sha256_file(hard_test_path),
            "model_sha256": sha256_file(MODEL_FILENAME),
            "metrics": {
                key: float(value)
                for key, value in selected_metrics.items()
                if key != "model"
            },
            "hard_test_error_count": int((selected_prediction != y_hard).sum()),
            "versions": {
                "python": platform.python_version(),
                "scikit_learn": sklearn.__version__,
                "pandas": pd.__version__,
                "joblib": joblib.__version__,
                "numpy": np.__version__,
            },
        }

        Path(METADATA_FILENAME).write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("Artifact tersimpan:", MODEL_FILENAME)
        print("Metadata tersimpan:", METADATA_FILENAME)
        display(pd.DataFrame([metadata["metrics"]]))
        """
    ),
    code(
        """
        # Khusus Google Colab: unduh kedua artifact.
        try:
            from google.colab import files

            files.download(MODEL_FILENAME)
            files.download(METADATA_FILENAME)
        except ImportError:
            print("Bukan Google Colab; artifact tersedia di working directory.")
        """
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {
            "name": "Training_Intent_13.ipynb",
            "provenance": [],
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT_PATH.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
    encoding="utf-8",
)
print(f"Wrote {OUTPUT_PATH}")
