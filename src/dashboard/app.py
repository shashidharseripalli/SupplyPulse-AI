"""SupplyPulse AI — Streamlit Interactive Resilience Cockpit.

Main entry point for interactive supply chain analytics, demand forecasting,
and resilience intelligence.
Run with: streamlit run src/dashboard/app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

from src.forecasting.evaluation import compute_all_metrics
from src.forecasting.lgbm_forecaster import LGBMDemandForecaster
from src.forecasting.seasonal_naive import SeasonalNaiveForecaster
from src.forecasting.xgb_forecaster import XGBDemandForecaster
from src.preprocessing.features import DemandFeaturePipeline


def render_home_page() -> None:
    """Render platform architecture and module overview."""
    st.title("📦 SupplyPulse AI")
    st.subheader(
        "Event-Aware Predictive Logistics & Supply Chain Resilience Platform"
    )
    st.markdown(
        """
        Welcome to **SupplyPulse AI**, an advanced decision support platform engineered
        to forecast demand volatilities, detect supplier disruptions, and optimize inventory
        levels under stochastic conditions.

        ---

        ### Platform Modules
        1. **📈 Demand Forecasting** *(Module 1 — Active)*:
           - Seasonal Naive statistical baseline
           - XGBoost gradient-boosted regression
           - LightGBM fast histogram boosting
           - Multi-metric evaluation (MAE, RMSE, WAPE, SMAPE)
        2. **📰 NLP Disruption Intelligence** *(Module 2)*:
           - Unstructured event classification & named entity extraction
        3. **⚠️ Supply Risk Prediction** *(Module 3)*:
           - Multi-class supplier risk scoring & isolation forest anomaly detection
        4. **📦 Inventory Optimization** *(Module 4)*:
           - Google OR-Tools multi-echelon safety stock & reorder policy
        5. **🔄 Scenario Simulation** *(Module 5)*:
           - Monte Carlo what-if simulation for demand surges and delivery delays
        """
    )


def generate_reference_series(
    product_id: str,
    days: int = 90,
    base_demand: float = 120.0,
    seasonality_period: int = 7,
) -> pd.DataFrame:
    """Generate a clean synthetic reference demand series for interactive demonstration."""
    np.random.seed(hash(product_id) % (2**31))
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")
    t = np.arange(days)

    trend = 0.25 * t
    seasonal = 25.0 * np.sin(2 * np.pi * t / seasonality_period)
    weekend_boost = np.where(pd.Series(dates).dt.dayofweek >= 5, 15.0, 0.0)
    noise = np.random.normal(0, 5, days)

    demand = np.maximum(5.0, base_demand + trend + seasonal + weekend_boost + noise)
    return pd.DataFrame({"date": dates, "demand": demand.round(1)})


def render_demand_forecast_page() -> None:
    """Render Module 1 interactive demand forecasting interface."""
    st.title("📈 Demand Forecasting Engine")
    st.caption("Module 1: Seasonal Naive, XGBoost, and LightGBM Models")

    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        product_id = st.selectbox(
            "Product / SKU",
            ["SKU-1001 (Consumer Tech)", "SKU-2002 (Industrial Parts)", "SKU-3003 (Pharma Supplies)"],
        )
    with col2:
        horizon = st.slider("Forecast Horizon (Days)", min_value=7, max_value=60, value=30, step=7)
    with col3:
        model_choice = st.selectbox(
            "Forecasting Model",
            ["Compare All Models", "Seasonal Naive", "XGBoost", "LightGBM"],
        )

    # Generate historical reference data
    df_hist = generate_reference_series(product_id, days=120)

    # Feature engineering for tree models
    pipeline = DemandFeaturePipeline(
        date_col="date",
        target_col="demand",
        lags=[1, 2, 7, 14, 21, 28],
        windows=[7, 14, 28],
        drop_na=True,
    )
    pipeline.fit(df_hist)
    df_feat = pipeline.transform(df_hist)

    X_train = df_feat[pipeline.feature_names_]
    y_train = df_feat["demand"]

    # Fit models
    naive_model = SeasonalNaiveForecaster(seasonal_period=7).fit(df_hist["demand"])
    naive_preds = naive_model.predict(horizon=horizon)

    xgb_model = XGBDemandForecaster(params={"n_estimators": 100, "max_depth": 5})
    xgb_model.fit(X_train, y_train)

    lgbm_model = LGBMDemandForecaster(params={"n_estimators": 100, "max_depth": 5})
    lgbm_model.fit(X_train, y_train)

    # Autoregressive multi-step rollout for tree models
    def rollout_tree_forecast(model) -> np.ndarray:
        curr = df_hist.copy()
        future: list[float] = []
        for _ in range(horizon):
            next_date = curr["date"].iloc[-1] + pd.Timedelta(days=1)
            f_df = pipeline.transform(curr)
            if len(f_df) == 0:
                p = float(curr["demand"].iloc[-1])
            else:
                last_row = f_df[pipeline.feature_names_].iloc[-1:]
                p = float(model.predict(last_row)[0])
            future.append(p)
            curr = pd.concat([curr, pd.DataFrame([{"date": next_date, "demand": p}])], ignore_index=True)
        return np.array(future)

    xgb_preds = rollout_tree_forecast(xgb_model)
    lgbm_preds = rollout_tree_forecast(lgbm_model)

    # Build forecast timeline dataframe
    future_dates = pd.date_range(
        start=df_hist["date"].iloc[-1] + pd.Timedelta(days=1),
        periods=horizon,
        freq="D",
    )

    plot_df = pd.DataFrame({"Date": df_hist["date"], "Historical Demand": df_hist["demand"]})

    if model_choice in ("Compare All Models", "Seasonal Naive"):
        plot_df_naive = pd.DataFrame({"Date": future_dates, "Seasonal Naive": naive_preds.round(1)})
        plot_df = pd.merge(plot_df, plot_df_naive, on="Date", how="outer")

    if model_choice in ("Compare All Models", "XGBoost"):
        plot_df_xgb = pd.DataFrame({"Date": future_dates, "XGBoost": xgb_preds.round(1)})
        plot_df = pd.merge(plot_df, plot_df_xgb, on="Date", how="outer")

    if model_choice in ("Compare All Models", "LightGBM"):
        plot_df_lgbm = pd.DataFrame({"Date": future_dates, "LightGBM": lgbm_preds.round(1)})
        plot_df = pd.merge(plot_df, plot_df_lgbm, on="Date", how="outer")

    plot_df = plot_df.set_index("Date")

    # Time series chart
    st.subheader("Forecast Trajectory")
    st.line_chart(plot_df)

    # Evaluation metrics comparison table
    st.subheader("Model Evaluation & Backtest Performance")
    in_sample_naive = SeasonalNaiveForecaster(seasonal_period=7).fit(df_hist["demand"].iloc[:-14]).predict(14)
    naive_metrics = compute_all_metrics(df_hist["demand"].iloc[-14:].to_numpy(), in_sample_naive)

    in_sample_xgb = xgb_model.predict(X_train.iloc[-14:])
    xgb_metrics = compute_all_metrics(y_train.iloc[-14:].to_numpy(), in_sample_xgb)

    in_sample_lgbm = lgbm_model.predict(X_train.iloc[-14:])
    lgbm_metrics = compute_all_metrics(y_train.iloc[-14:].to_numpy(), in_sample_lgbm)

    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric("Seasonal Naive WAPE", f"{naive_metrics['wape']:.2f}%", help="Weighted Absolute Percentage Error")
        st.metric("Seasonal Naive MAE", f"{naive_metrics['mae']:.2f}")
    with m_col2:
        st.metric("XGBoost WAPE", f"{xgb_metrics['wape']:.2f}%", delta=f"{xgb_metrics['wape'] - naive_metrics['wape']:.2f}% (vs Naive)")
        st.metric("XGBoost MAE", f"{xgb_metrics['mae']:.2f}")
    with m_col3:
        st.metric("LightGBM WAPE", f"{lgbm_metrics['wape']:.2f}%", delta=f"{lgbm_metrics['wape'] - naive_metrics['wape']:.2f}% (vs Naive)")
        st.metric("LightGBM MAE", f"{lgbm_metrics['mae']:.2f}")

    # Feature Importance Section
    if model_choice in ("XGBoost", "LightGBM", "Compare All Models"):
        st.subheader("Feature Importances")
        feat_col1, feat_col2 = st.columns(2)
        with feat_col1:
            st.caption("XGBoost Top Features")
            st.bar_chart(xgb_model.get_feature_importances().head(8))
        with feat_col2:
            st.caption("LightGBM Top Features")
            st.bar_chart(lgbm_model.get_feature_importances().head(8))


def main() -> None:
    """Main dashboard entry point."""
    st.set_page_config(
        page_title="SupplyPulse AI",
        page_icon="📦",
        layout="wide",
    )

    st.sidebar.title("📦 SupplyPulse AI")
    st.sidebar.caption("Logistics & Resilience Cockpit")

    page = st.sidebar.radio(
        "Navigation",
        [
            "🏠 Home",
            "📈 Demand Forecast",
            "⚠️ Risk Assessment",
            "📦 Inventory",
            "🔄 Simulation",
        ],
    )

    if page == "🏠 Home":
        render_home_page()
    elif page == "📈 Demand Forecast":
        render_demand_forecast_page()
    elif page == "⚠️ Risk Assessment":
        st.title("⚠️ Risk Assessment")
        st.info("Module 3: Risk prediction is scheduled next.")
    elif page == "📦 Inventory":
        st.title("📦 Inventory Optimization")
        st.info("Module 4: Inventory optimization using Google OR-Tools.")
    elif page == "🔄 Simulation":
        st.title("🔄 Scenario Simulation")
        st.info("Module 5: Stochastic disruption scenario engine.")


if __name__ == "__main__":
    main()
