import numpy as np
from sklearn.linear_model import LogisticRegression


class LogisticRegressionPredictor:
    def __init__(self) -> None:
        self.model = LogisticRegression(max_iter=5000)

    def predict_signal(self, features: dict) -> dict:
        vector = np.array(
            [
                features.get("return_1", 0.0),
                features.get("return_5", 0.0),
                features.get("return_20", 0.0),
                features.get("ma_spread", 0.0),
                features.get("rsi", 50.0),
                features.get("macd", 0.0),
                features.get("volume_change", 0.0),
                features.get("recent_change", 0.0),
            ],
            dtype=float,
        ).reshape(1, -1)

        # Use a lightweight deterministic rule when the model is not trained.
        if not hasattr(self.model, "classes_"):
            score = float(vector[0, 0] + vector[0, 4] / 100 + vector[0, 7] / 100)
            probability = 0.5 + min(0.45, max(-0.45, score / 10))
            direction = "bullish" if probability >= 0.5 else "bearish"
            return {
                "direction": direction,
                "confidence": round(abs(probability - 0.5) * 200, 1),
                "probability": round(probability, 3),
            }

        probability = float(self.model.predict_proba(vector)[0, 1])
        direction = "bullish" if probability >= 0.5 else "bearish"
        return {
            "direction": direction,
            "confidence": round(abs(probability - 0.5) * 200, 1),
            "probability": round(probability, 3),
        }
