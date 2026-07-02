"""
ml.py
-----
Loads the trained model + scaler once at import time and exposes a single
`predict_flood()` function used by the Flask routes. Keeping this separate
from routes.py keeps the ML integration clean and easy to swap out later
(e.g. for a different model format or a remote inference service).
"""

import json
import os

import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "flood_model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")


class ModelNotReadyError(RuntimeError):
    """Raised when the trained model artifacts are missing."""


_model = None
_scaler = None
_metadata = None


def _load_artifacts():
    global _model, _scaler, _metadata

    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise ModelNotReadyError(
            "Trained model not found. Run 'python train.py' first to generate "
            "models/flood_model.joblib and models/scaler.joblib."
        )

    _model = joblib.load(MODEL_PATH)
    _scaler = joblib.load(SCALER_PATH)

    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as f:
            _metadata = json.load(f)
    else:
        _metadata = {"feature_order": None, "best_model_name": "Unknown"}


def get_feature_order():
    if _metadata is None:
        _load_artifacts()
    order = _metadata.get("feature_order")
    if not order:
        # Fallback: matches train.py's FEATURE_ORDER
        order = [
            "Temp", "Humidity", "Cloud Cover", "ANNUAL",
            "Jan-Feb", "Mar-May", "Jun-Sep", "Oct-Dec", "avgjune", "sub",
        ]
    return order


def get_model_name():
    if _metadata is None:
        _load_artifacts()
    return _metadata.get("best_model_name", "Unknown")


def get_metadata():
    if _metadata is None:
        _load_artifacts()
    return _metadata


def predict_flood(feature_dict):
    """Run inference on a single sample.

    Args:
        feature_dict: dict mapping feature name -> numeric value. Must
            contain every key returned by get_feature_order().

    Returns:
        dict with prediction (0/1), flood_probability, no_flood_probability,
        and a human-readable message.
    """
    if _model is None or _scaler is None:
        _load_artifacts()

    feature_order = get_feature_order()
    missing = [f for f in feature_order if f not in feature_dict]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    try:
        values = [float(feature_dict[f]) for f in feature_order]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"All feature values must be numeric: {exc}") from exc

    input_array = pd.DataFrame([values], columns=feature_order)
    input_scaled = _scaler.transform(input_array)

    prediction = int(_model.predict(input_scaled)[0])
    proba = _model.predict_proba(input_scaled)[0]

    flood_probability = float(proba[1])
    no_flood_probability = float(proba[0])

    if prediction == 1:
        message = "High flood risk detected. Consider issuing an alert."
        risk_level = "high" if flood_probability >= 0.75 else "moderate"
    else:
        message = "Low flood risk. Conditions appear safe."
        risk_level = "low"

    return {
        "prediction": prediction,
        "flood_probability": round(flood_probability * 100, 2),
        "no_flood_probability": round(no_flood_probability * 100, 2),
        "message": message,
        "risk_level": risk_level,
        "model_name": get_model_name(),
    }
