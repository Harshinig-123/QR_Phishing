"""
ml_predict.py
-------------
Stage 3: Inference and Local Explanation Interface for URL Phishing Classification.

Features & Contract:
    predict_url(url: str) -> dict
    should_run_stage3(stage1_result: Any) -> bool

Output Schema:
    {
        "url": str,
        "prediction": "phishing" | "legitimate",
        "probability": float,           # confidence of the predicted class (0.0000 - 1.0000)
        "phishing_probability": float,  # probability of phishing (0.0000 - 1.0000)
        "top_factors": [                # top 5 per-URL linear contributions from LR
            {
                "feature": str,
                "value": Any,
                "direction": "phishing" | "legitimate",
                "contribution": float
            }
        ],
        "features": dict                # the extracted 23 static URL features
    }

Rationale for `top_factors` Formulation:
    While the Random Forest provides global feature importance scores across the entire
    corpus, Logistic Regression with standardized features provides an exact, instance-level
    linear contribution:
        contribution_i = coef_i * scaled_feature_i
    A positive contribution pushes the log-odds toward 'phishing', while a negative contribution
    pushes toward 'legitimate'. Ranking features by |contribution_i| yields an interpretable,
    verifiable local explanation for why this specific URL was flagged or cleared.

CLI Usage:
    python ml_predict.py "https://example.com"
    python ml_predict.py "http://192.168.1.50:8080/login/update"
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from urllib.parse import urlsplit

import joblib
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"
MODEL_PATH = MODELS_DIR / "phishing_model.joblib"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from feature_extractor import extract_ml_features, FEATURE_ORDER

# Global cache for lazy model loading
_MODEL_CACHE: Optional[Dict[str, Any]] = None


def get_model_bundle() -> Dict[str, Any]:
    """
    Lazy-loads and caches the trained model bundle.
    Raises FileNotFoundError with clear instructions if train_model.py has not been run.
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


def should_run_stage3(stage1_result: Any) -> bool:
    """
    Determines whether Stage 3 (ML Classification) should be executed based on
    the decoded payload from Stage 1.

    Returns True ONLY if Stage 1 successfully decoded and validated an HTTP(S) URL.
    Returns False for non-URL payloads (WiFi, phone/SMS, payment, vCard, plain text).

    Args:
        stage1_result: Either a Stage 1 QRDecodeResult object, a dict representation,
                       or a raw payload string.
    """
    if stage1_result is None:
        return False

    status = None
    url = None
    raw_data = None
    payload_type = None

    # Handle QRDecodeResult dataclass or object
    if hasattr(stage1_result, "status"):
        status = getattr(stage1_result, "status")
        url = getattr(stage1_result, "url", None)
        raw_data = getattr(stage1_result, "raw_data", None)
        payload_type = getattr(stage1_result, "payload_type", None)
    elif isinstance(stage1_result, dict):
        status = stage1_result.get("status")
        url = stage1_result.get("url")
        raw_data = stage1_result.get("raw_data")
        payload_type = stage1_result.get("payload_type")
    elif isinstance(stage1_result, str):
        raw_data = stage1_result
        url = stage1_result

    # If payload type is explicitly recorded
    if payload_type and str(payload_type).lower() != "url":
        return False

    # Check non-URL scheme prefixes
    raw_text = (raw_data or url or "").strip()
    non_url_prefixes = (
        "wifi:", "smsto:", "tel:", "sms:", "mailto:", "begin:vcard",
        "upi:", "pay:", "geo:", "bitcoin:", "market:"
    )
    if any(raw_text.lower().startswith(p) for p in non_url_prefixes):
        return False

    # Check Stage 1 status contract
    if status is not None and status != "ok":
        return False

    # Check URL validity (must be http/https)
    target = url or raw_text
    try:
        parsed = urlsplit(target)
        if parsed.scheme.lower() in ("http", "https") and parsed.netloc:
            return True
    except Exception:
        return False

    return False


def predict_url(url: str) -> Dict[str, Any]:
    """
    Predicts whether a URL is phishing or legitimate using the trained
    Random Forest model and produces local feature explanations via the
    standardized Logistic Regression model.

    Guarantees:
      - 100% static computation with ZERO network access.
      - Expects that Stage 2 has already resolved shorteners, so `url` is
        the final destination string.
    """
    bundle = get_model_bundle()
    rf_model = bundle["model"]
    lr_model = bundle["lr_explanation_model"]
    scaler = bundle["scaler"]
    feature_order = bundle.get("feature_order", FEATURE_ORDER)
    label_mapping = bundle.get("label_mapping", {1: "phishing", 0: "legitimate"})

    # 1. Extract 23 static URL features
    features_dict = extract_ml_features(url)
    feature_vector = [float(features_dict[f]) for f in feature_order]
    X_sample = np.array([feature_vector], dtype=float)

    # 2. Random Forest Primary Prediction
    rf_proba = rf_model.predict_proba(X_sample)[0]
    rf_classes = list(rf_model.classes_)
    phishing_class_idx = 1
    phish_loc = rf_classes.index(phishing_class_idx) if phishing_class_idx in rf_classes else 1
    phishing_prob = float(rf_proba[phish_loc])

    # Class assignment threshold (0.50)
    if phishing_prob >= 0.50:
        pred_label = 1
        confidence = phishing_prob
    else:
        pred_label = 0
        confidence = 1.0 - phishing_prob

    prediction_str = label_mapping.get(pred_label, "phishing" if pred_label == 1 else "legitimate")

    # 3. Local Feature Contributions (via Scaled Logistic Regression)
    X_scaled = scaler.transform(X_sample)[0]
    lr_coefs = lr_model.coef_[0]

    contributions = []
    for f_name, f_val, scaled_val, coef in zip(feature_order, feature_vector, X_scaled, lr_coefs):
        contrib = float(coef * scaled_val)
        contributions.append({
            "feature": f_name,
            "value": f_val,
            "direction": "phishing" if contrib > 0 else "legitimate",
            "contribution": round(abs(contrib), 4),
            "_raw_contrib": contrib
        })

    # Sort by absolute contribution magnitude descending for the top 5 factors
    contributions.sort(key=lambda x: abs(x["_raw_contrib"]), reverse=True)
    top_5_factors = [
        {
            "feature": c["feature"],
            "value": c["value"],
            "direction": c["direction"],
            "contribution": c["contribution"],
        }
        for c in contributions[:5]
    ]

    return {
        "url": url,
        "prediction": prediction_str,
        "probability": round(confidence, 4),
        "phishing_probability": round(phishing_prob, 4),
        "top_factors": top_5_factors,
        "features": features_dict,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python ml_predict.py \"<url>\"")
        print("Example: python ml_predict.py \"https://www.google.com\"")
        print("Example: python ml_predict.py \"http://192.168.1.50/login/update\"")
        sys.exit(1)

    target_url = sys.argv[1]
    result = predict_url(target_url)

    print(json.dumps(result, indent=2))
    print("\nSummary:")
    print(f"  Target URL           : {result['url']}")
    print(f"  Prediction           : {result['prediction'].upper()}")
    print(f"  Confidence           : {result['probability'] * 100:.2f}%")
    print(f"  Phishing Probability : {result['phishing_probability'] * 100:.2f}%")
    print("\nTop Contributing Factors:")
    for i, factor in enumerate(result["top_factors"], 1):
        print(f"  {i}. {factor['feature']} = {factor['value']} -> favors {factor['direction'].upper()} (influence: {factor['contribution']})")


if __name__ == "__main__":
    main()
