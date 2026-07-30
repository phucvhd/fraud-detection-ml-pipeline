"""Shared feature engineering for the fraud-detection pipeline.

This is the single source of truth for turning raw transaction columns
(``Time``, ``V1``..``V28``, ``Amount``) into the model's feature matrix. It is
used by training, evaluation, hyper-parameter tuning AND the SageMaker inference
script so that the exact same transformation is applied everywhere — this is what
prevents train/serve skew.

The module intentionally depends only on numpy / pandas / scikit-learn so it can
be bundled as-is inside the model ``.tar.gz`` and imported by the inference
container without pulling in the rest of ``src``.

Fitted state (the ``Amount`` scaler, the training ``Time`` maximum used to
normalise ``time_since_start``, and which feature groups are enabled) is captured
in a ``preprocessor`` dict at ``fit`` time and reused verbatim afterwards. Never
re-fit a scaler on evaluation or serving data.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def engineer_features(df: pd.DataFrame, config: dict = None,
                      preprocessor: dict = None, fit: bool = False):
    """Build the model feature matrix from raw transaction columns.

    Args:
        df: Raw features (``Class`` already dropped by the caller).
        config: Full pipeline config. Required when ``fit=True`` — the enabled
            feature groups are read from ``config['features']``.
        preprocessor: Fitted state from a previous ``fit=True`` call. Required
            when ``fit=False``.
        fit: Fit the scaler / time normaliser on ``df`` and return the populated
            ``preprocessor``. Use ``True`` on the training split only.

    Returns:
        ``(X, preprocessor)`` where ``X`` is the engineered feature frame and
        ``preprocessor`` carries the fitted state to reuse downstream.
    """
    X = df.copy()

    if fit:
        if config is None:
            raise ValueError("config is required when fit=True")
        preprocessor = {
            "flags": {
                "time": bool(config["features"]["time_features"]["enabled"]),
                "amount": bool(config["features"]["amount_features"]["enabled"]),
            }
        }
    else:
        if preprocessor is None:
            raise ValueError("preprocessor is required when fit=False")

    flags = preprocessor["flags"]

    if flags["time"]:
        X["hour_of_day"] = (X["Time"] / 3600) % 24

        # Cast the binned period to int: RandomForest cannot consume a pandas
        # Categorical dtype, and the category codes must be stable across runs.
        X["day_period"] = pd.cut(
            X["hour_of_day"], bins=[0, 6, 12, 18, 24],
            labels=[0, 1, 2, 3], include_lowest=True
        ).astype(int)

        if fit:
            preprocessor["time_max"] = float(X["Time"].max())
        # Normalise against the training-time maximum so serving records are
        # scaled on the same basis as training (a per-batch max would skew).
        time_max = preprocessor.get("time_max") or 1.0
        X["time_since_start"] = X["Time"] / time_max

    if flags["amount"]:
        X["log_amount"] = np.log1p(X["Amount"])

        if fit:
            scaler = StandardScaler()
            X["amount_scaled"] = scaler.fit_transform(X[["Amount"]])
            preprocessor["amount_scaler"] = scaler
        else:
            scaler = preprocessor["amount_scaler"]
            X["amount_scaled"] = scaler.transform(X[["Amount"]])

    return X, preprocessor
