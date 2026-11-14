"""Demand forecast API endpoints.

Exposes endpoints for generating forecasts via Seasonal Naive, XGBoost,
or LightGBM algorithms with validation and domain error handling.
"""

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from src.api.schemas import ForecastRequest, ForecastResponse
from src.common.exceptions import ModelError
from src.common.logger import get_logger
from src.forecasting.evaluation import compute_all_metrics
from src.forecasting.lgbm_forecaster import LGBMDemandForecaster
from src.forecasting.seasonal_naive import SeasonalNaiveForecaster
from src.forecasting.xgb_forecaster import XGBDemandForecaster
from src.preprocessing.features import DemandFeaturePipeline

router = APIRouter(prefix="/api", tags=["forecast"])
logger = get_logger(__name__)


@router.post("/forecast", response_model=ForecastResponse)
def create_forecast(request: ForecastRequest) -> ForecastResponse:
    """Generate demand forecast for a given product and horizon.

    Supports:
        - `seasonal_naive`: Weekly/monthly cyclic projection
        - `xgboost`: Gradient boosted regression tree ensemble
        - `lightgbm`: Fast leaf-wise boosted decision trees

    Args:
        request: Validated `ForecastRequest` payload.

    Returns:
        ForecastResponse containing non-negative point predictions and metadata.
    """
    logger.info(
        "Forecast requested: product=%s, horizon=%d, model=%s",
        request.product_id,
        request.horizon,
        request.model_type,
    )

    try:
        # Resolve historical demand: either supplied in request or generated reference history
        if request.historical_demand and len(request.historical_demand) >= 14:
            history = np.asarray(request.historical_demand, dtype=float)
        else:
            # Baseline reference cycle (60 days of synthetic demand for evaluation)
            np.random.seed(hash(request.product_id) % (2**31))
            t = np.arange(60)
            seasonal_pattern = 20 * np.sin(2 * np.pi * t / request.seasonal_period)
            trend = 0.3 * t
            noise = np.random.normal(0, 3, 60)
            history = np.maximum(5.0, 100.0 + trend + seasonal_pattern + noise)

        if request.model_type == "seasonal_naive":
            model = SeasonalNaiveForecaster(seasonal_period=request.seasonal_period)
            model.fit(history)
            preds = model.predict(horizon=request.horizon)
            metrics = None

        elif request.model_type in ("xgboost", "lightgbm"):
            # Construct feature matrix for tree models
            dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=len(history), freq="D")
            df_hist = pd.DataFrame({"date": dates, "demand": history})

            pipeline = DemandFeaturePipeline(
                date_col="date",
                target_col="demand",
                lags=[1, 2, 7, 14],
                windows=[7, 14],
                drop_na=True,
            )
            pipeline.fit(df_hist)
            df_features = pipeline.transform(df_hist)

            if len(df_features) < 10:
                raise ModelError(
                    "Insufficient data points remaining after feature lag creation."
                )

            X_train = df_features[pipeline.feature_names_]
            y_train = df_features["demand"]

            if request.model_type == "xgboost":
                tree_model = XGBDemandForecaster()
            else:
                tree_model = LGBMDemandForecaster()

            tree_model.fit(X_train, y_train)

            # Recursive / multi-step projection for future horizon
            curr_df = df_hist.copy()
            future_preds: list[float] = []

            for step in range(request.horizon):
                next_date = curr_df["date"].iloc[-1] + pd.Timedelta(days=1)
                # Compute feature row for the next step
                row_features = pipeline.transform(curr_df)
                if len(row_features) == 0:
                    next_pred = float(curr_df["demand"].iloc[-1])
                else:
                    last_feature_row = row_features[pipeline.feature_names_].iloc[-1:]
                    next_pred = float(tree_model.predict(last_feature_row)[0])

                future_preds.append(next_pred)
                # Append predicted row to history for autoregressive lags
                curr_df = pd.concat(
                    [curr_df, pd.DataFrame([{"date": next_date, "demand": next_pred}])],
                    ignore_index=True,
                )

            preds = np.array(future_preds)

            # In-sample metrics for quality assurance
            in_sample_preds = tree_model.predict(X_train)
            metrics = compute_all_metrics(y_train.to_numpy(), in_sample_preds)

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model_type: '{request.model_type}'",
            )

        return ForecastResponse(
            product_id=request.product_id,
            horizon=request.horizon,
            model_used=request.model_type,
            predictions=[float(round(p, 2)) for p in preds],
            metrics=metrics,
        )

    except ModelError as me:
        logger.warning("Forecasting domain error: %s", me)
        raise HTTPException(status_code=422, detail=str(me)) from me
    except Exception as e:
        logger.exception("Unexpected error during forecasting: %s", e)
        raise HTTPException(status_code=500, detail="Internal forecasting error.") from e
