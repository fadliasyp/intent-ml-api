import unittest

import app


class IntentApiTest(unittest.TestCase):
    def test_model_uses_complete_intent_contract(self):
        self.assertEqual(set(app.model_classes), app.EXPECTED_INTENTS)
        self.assertEqual(len(app.model_classes), 13)

    def test_health_reports_loaded_model(self):
        payload = app.health()
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["model_loaded"])
        self.assertEqual(payload["intent_count"], 13)

    def test_prediction_response_keeps_chatbot_contract(self):
        result = app.predict_intent(
            app.PredictRequest(question="metode pembayarannya apa aja")
        )

        required_fields = {
            "intent",
            "confidence",
            "top3",
            "method",
            "is_low_confidence",
            "model_name",
        }
        self.assertTrue(required_fields.issubset(result))
        self.assertEqual(result["intent"], "shipping_transaction")
        self.assertEqual(len(result["top3"]), 3)
        self.assertGreaterEqual(result["confidence"], 0)
        self.assertLessEqual(result["confidence"], 1)

    def test_critical_intent_examples(self):
        cases = {
            "cara membeli produk ini gimana": "shipping_transaction",
            "bandingkan chogokin A dengan chogokin B": "compare",
            "paket jne saya sekarang ada di kota mana": "shipment_tracking",
            "produk di bawah 1 juta apa saja": "price_promo",
            "apakah full diecast": "product_detail",
            "jam berapa sekarang": "general",
        }

        for question, expected_intent in cases.items():
            with self.subTest(question=question):
                result = app.predict_intent(
                    app.PredictRequest(question=question)
                )
                self.assertEqual(result["intent"], expected_intent)

    def test_whitespace_only_question_is_rejected(self):
        with self.assertRaises(app.HTTPException) as context:
            app.predict_intent(app.PredictRequest(question="   "))

        self.assertEqual(context.exception.status_code, 400)

    def test_incomplete_metadata_intent_contract_is_rejected(self):
        invalid_metadata = {
            **app.metadata,
            "labels": ["general"],
        }

        with self.assertRaisesRegex(RuntimeError, "metadata"):
            app.load_and_validate_model(app.MODEL_PATH, invalid_metadata)


if __name__ == "__main__":
    unittest.main()
