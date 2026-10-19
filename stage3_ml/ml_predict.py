"""
ml_predict.py
-------------
Stage 3: Inference and Prediction Interface for URL Phishing Classification.

Exposes:
    predict_url(url: str) -> dict

Contract:
    Returns {
        "url": str,
        "prediction": "phishing" | "legitimate",
        "probability": float,   # confidence of the predicted class (0.0000 - 1.0000)
        "features": dict        # the extracted 10 URL-computable features
    }

CLI Usage:
    python ml_predict.py "https://example.com"
    python ml_predict.py "http://192.168.1.50/login/update"
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import joblib
import numpy as np

# Define paths relative to this file
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "models" / "phishing_model.joblib"

# Import feature extractor
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from feature_extractor import extract_ml_features, FEATURE_ORDER

# Global cache for lazy model loading
_MODEL_CACHE: Optional[Dict[str, Any]] = None


def get_model_bundle() -> Dict[str, Any]:
    """
    Lazy-loads and caches the trained model bundle.
    Raises a clear FileNotFoundError if train_model.py has not been run.
    """
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{MODEL_PATH}'. "
            "Please train the model first by executing: python stage3_ml/train_model.py"
        )

    try:
        bundle = joblib.load(MODEL_PATH)
        _MODEL_CACHE = bundle
        return bundle
    except Exception as e:
        raise RuntimeError(f"Failed to load model file from '{MODEL_PATH}': {e}") from e


def predict_url(url: str) -> Dict[str, Any]:
    """
    Extracts URL-computable features and predicts whether the URL is phishing
    or legitimate using the trained machine learning model.

    Args:
        url: The candidate URL string (typically the resolved_url from Stage 2).

    Returns:
        dict: {
            "url": str,
            "prediction": "phishing" | "legitimate",
            "probability": float,
            "features": Dict[str, int]
        }
    """
    bundle = get_model_bundle()
    model = bundle["model"]
    feature_order = bundle.get("feature_order", FEATURE_ORDER)
    label_mapping = bundle.get("label_mapping", {1: "phishing", 0: "legitimate"})

    # Extract 10 URL-computable features
    features_dict = extract_ml_features(url)
    feature_vector = [features_dict[f] for f in feature_order]

    # Model inference
    feat_arr = np.array([feature_vector])
    proba = model.predict_proba(feat_arr)[0]
    classes = list(model.classes_)

    # Identify predicted class index
    pred_class_idx = int(model.predict(feat_arr)[0])
    class_loc = classes.index(pred_class_idx)
    confidence = float(proba[class_loc])

    prediction_str = label_mapping.get(pred_class_idx, "phishing" if pred_class_idx == 1 else "legitimate")

    return {
        "url": url,
        "prediction": prediction_str,
        "probability": round(confidence, 4),
        "features": features_dict,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python ml_predict.py \"<url>\"")
        print("Example: python ml_predict.py \"http://192.168.1.1/admin\"")
        sys.exit(1)

    target_url = sys.argv[1]
    result = predict_url(target_url)

    print(json.dumps(result, indent=2))
    print("\nSummary:")
    print(f"  Target URL : {result['url']}")
    print(f"  Prediction : {result['prediction'].upper()}")
    print(f"  Confidence : {result['probability'] * 100:.2f}%")


if __name__ == "__main__":
    main()
