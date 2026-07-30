"""Tests for the shared feature-engineering module.

These guard the train/serve-skew fix: the same transformation must run at fit
and transform time, the fitted scaler / time-normaliser must be reused rather
than refit, and the engineered column set must be stable.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.features.feature_engineering import engineer_features

CONFIG = {
    "features": {
        "time_features": {"enabled": True},
        "amount_features": {"enabled": True},
    }
}

EXPECTED_ENGINEERED = {
    "hour_of_day", "day_period", "time_since_start", "log_amount", "amount_scaled"
}


def _make_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    data = {"Time": rng.uniform(0, 172800, n)}
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(size=n)
    data["Amount"] = rng.exponential(scale=88.0, size=n)
    return pd.DataFrame(data)


def test_fit_adds_expected_columns():
    df = _make_df()
    x, preprocessor = engineer_features(df, config=CONFIG, fit=True)

    assert EXPECTED_ENGINEERED.issubset(set(x.columns))
    assert "amount_scaler" in preprocessor
    assert "time_max" in preprocessor
    assert preprocessor["flags"] == {"time": True, "amount": True}


def test_day_period_is_integer_not_categorical():
    df = _make_df()
    x, _ = engineer_features(df, config=CONFIG, fit=True)

    # RandomForest cannot consume a pandas Categorical dtype.
    assert pd.api.types.is_integer_dtype(x["day_period"])
    assert set(x["day_period"].unique()).issubset({0, 1, 2, 3})


def test_transform_reuses_fitted_scaler_not_refit():
    train_df = _make_df(seed=1)
    _, preprocessor = engineer_features(train_df, config=CONFIG, fit=True)

    # A serving batch with a very different Amount distribution must be scaled
    # using the TRAINING scaler, not a scaler refit on the serving batch.
    serve_df = _make_df(n=10, seed=99)
    serve_df["Amount"] = serve_df["Amount"] * 1000

    x_serve, _ = engineer_features(serve_df, preprocessor=preprocessor, fit=False)

    scaler = preprocessor["amount_scaler"]
    expected = scaler.transform(serve_df[["Amount"]]).ravel()
    np.testing.assert_allclose(x_serve["amount_scaled"].to_numpy(), expected)


def test_fit_and_transform_produce_identical_column_order():
    df = _make_df()
    x_fit, preprocessor = engineer_features(df, config=CONFIG, fit=True)
    x_transform, _ = engineer_features(df, preprocessor=preprocessor, fit=False)

    assert list(x_fit.columns) == list(x_transform.columns)


def test_transform_requires_preprocessor():
    df = _make_df()
    with pytest.raises(ValueError):
        engineer_features(df, fit=False)


def test_fit_requires_config():
    df = _make_df()
    with pytest.raises(ValueError):
        engineer_features(df, fit=True)


def test_disabled_feature_groups_are_skipped():
    df = _make_df()
    config = {
        "features": {
            "time_features": {"enabled": False},
            "amount_features": {"enabled": False},
        }
    }
    x, _ = engineer_features(df, config=config, fit=True)
    assert not EXPECTED_ENGINEERED.intersection(set(x.columns))
