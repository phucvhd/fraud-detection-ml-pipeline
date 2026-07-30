"""SageMaker inference handlers for the fraud-detection Random Forest model.

The model artifact bundle (``model.tar.gz``) ships three things this script
relies on, produced by ``src/deploy/package_model.py``:

* ``model.joblib``          – the trained RandomForestClassifier
* ``preprocessor.joblib``   – the fitted scaler / time-normaliser + feature flags
* ``feature_engineering.py`` – the SAME module used at training time

Because the identical ``engineer_features`` transformation runs here, the served
feature vector matches the trained one exactly (no train/serve skew).
"""

import json
import os
import sys

import joblib
import pandas as pd

# The bundle unpacks flat into ``model_dir``; make the bundled feature module importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import engineer_features  # noqa: E402

RAW_COLUMNS = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)


def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    preprocessor = joblib.load(os.path.join(model_dir, "preprocessor.joblib"))
    return {"model": model, "preprocessor": preprocessor}


def input_fn(request_body, content_type):
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")

    data = json.loads(request_body)
    # Accept a single record or a list of records.
    records = data if isinstance(data, list) else [data]
    return pd.DataFrame(records, columns=RAW_COLUMNS)


def predict_fn(input_data, artifacts):
    model = artifacts["model"]
    preprocessor = artifacts["preprocessor"]

    features, _ = engineer_features(input_data, preprocessor=preprocessor, fit=False)

    predictions = model.predict(features)
    results = []
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[:, 1]
        for pred, proba in zip(predictions, probabilities):
            results.append({
                "prediction": int(pred),
                "fraud_probability": float(proba),
                "is_fraud": bool(pred == 1),
            })
    else:
        for pred in predictions:
            results.append({"prediction": int(pred)})

    # Preserve the single-record convenience shape.
    return results[0] if len(results) == 1 else results


def output_fn(prediction, accept):
    if accept == "application/json":
        return json.dumps(prediction), accept
    raise ValueError(f"Unsupported accept type: {accept}")
