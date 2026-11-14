"""SupplyPulse AI — Streamlit Dashboard.

Main entry point for the interactive dashboard.
Run with: streamlit run src/dashboard/app.py
"""
import streamlit as st


def main() -> None:
    """Main dashboard application."""
    st.set_page_config(
        page_title="SupplyPulse AI",
        page_icon="📦",
        layout="wide",
    )

    # Sidebar navigation
    st.sidebar.title("📦 SupplyPulse AI")
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "📈 Demand Forecast", "⚠️ Risk Assessment",
         "📦 Inventory", "🔄 Simulation"],
    )

    if page == "🏠 Home":
        st.title("SupplyPulse AI")
        st.subheader("Event-Aware Predictive Logistics & Supply Chain Resilience Platform")
        st.markdown(
            """
            ### Modules
            1. **Demand Forecasting** — Seasonal Naive, XGBoost, LightGBM
            2. **NLP Disruption Intelligence** — Text classification & NER
            3. **Supply Risk Prediction** — Classification & anomaly detection
            4. **Inventory Optimization** — Google OR-Tools
            5. **Scenario Simulation** — What-if analysis

            Select a module from the sidebar to get started.
            """
        )
    elif page == "📈 Demand Forecast":
        st.title("📈 Demand Forecast")
        st.info("Module 1: Demand forecasting is not yet implemented.")
    elif page == "⚠️ Risk Assessment":
        st.title("⚠️ Risk Assessment")
        st.info("Module 3: Risk prediction is not yet implemented.")
    elif page == "📦 Inventory":
        st.title("📦 Inventory Optimization")
        st.info("Module 4: Inventory optimization is not yet implemented.")
    elif page == "🔄 Simulation":
        st.title("🔄 Scenario Simulation")
        st.info("Module 5: Scenario simulation is not yet implemented.")


if __name__ == "__main__":
    main()
