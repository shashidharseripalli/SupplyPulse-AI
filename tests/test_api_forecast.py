"""Integration tests for the forecast API endpoint."""

import pytest
from src.api.routes.forecast import create_forecast
from src.api.schemas import ForecastRequest


def test_api_create_forecast_seasonal_naive() -> None:
    """API endpoint should return valid seasonal naive predictions."""
    req = ForecastRequest(
        product_id="SKU-TEST-01",
        horizon=7,
        model_type="seasonal_naive",
        seasonal_period=7,
        historical_demand=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0] * 2,
    )
    resp = create_forecast(req)
    assert resp.product_id == "SKU-TEST-01"
    assert resp.horizon == 7
    assert resp.model_used == "seasonal_naive"
    assert len(resp.predictions) == 7
    assert resp.predictions == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]


def test_api_create_forecast_xgboost() -> None:
    """API endpoint should return valid predictions for XGBoost model."""
    req = ForecastRequest(
        product_id="SKU-TEST-02",
        horizon=5,
        model_type="xgboost",
    )
    resp = create_forecast(req)
    assert resp.product_id == "SKU-TEST-02"
    assert resp.horizon == 5
    assert resp.model_used == "xgboost"
    assert len(resp.predictions) == 5
    assert all(p >= 0.0 for p in resp.predictions)
    assert resp.metrics is not None
    assert "wape" in resp.metrics


def test_api_create_forecast_lightgbm() -> None:
    """API endpoint should return valid predictions for LightGBM model."""
    req = ForecastRequest(
        product_id="SKU-TEST-03",
        horizon=5,
        model_type="lightgbm",
    )
    resp = create_forecast(req)
    assert resp.product_id == "SKU-TEST-03"
    assert resp.horizon == 5
    assert resp.model_used == "lightgbm"
    assert len(resp.predictions) == 5
    assert all(p >= 0.0 for p in resp.predictions)
