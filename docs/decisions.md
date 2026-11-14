# Architecture Decision Records (ADR)

This document records every major architectural decision for SupplyPulse AI.

---

## ADR-001: Modular Directory Structure

**Status:** Accepted  
**Date:** 2024-11-14

**Context:**  
The platform comprises five distinct analytical modules plus API and dashboard layers. A monolithic structure would make testing, debugging, and future extension difficult for a solo developer.

**Decision:**  
Organize code into separate directories per analytical domain (`forecasting/`, `nlp/`, `risk/`, `optimization/`, `simulation/`), with shared utilities in `common/` and interface layers in `api/` and `dashboard/`.

**Consequences:**  
- Each module can be developed and tested in isolation.
- Clear ownership boundaries make code reviews and navigation straightforward.
- Slight overhead from more files, but manageable for this project scale.

---

## ADR-002: YAML Configuration over Hardcoded Settings

**Status:** Accepted  
**Date:** 2024-11-14

**Context:**  
Model hyperparameters, database URLs, API settings, and feature flags need to be easily adjustable without code changes.

**Decision:**  
Use YAML configuration files (`config/settings.yaml`) with environment variable interpolation support (`${VAR:default}`). Sensitive values go in `.env` (git-ignored).

**Consequences:**  
- Single source of truth for all tunables.
- Environment-specific overrides via env vars or `.env`.
- Requires a config loader utility but avoids scattered magic numbers.

---

## ADR-003: CPU-First Design

**Status:** Accepted  
**Date:** 2024-11-14

**Context:**  
The development machine has an AMD Radeon GPU (no CUDA). The project is an academic prototype that must run reliably on any machine.

**Decision:**  
All models default to CPU execution. No CUDA/cuDNN dependencies. GPU acceleration can be added later as an optional optimization.

**Consequences:**  
- Guaranteed portability across development and evaluation machines.
- Slower training on large datasets, but acceptable for prototype scale.
- NLP models use DistilBERT (smaller, CPU-friendly) rather than full BERT.

---

## ADR-004: MLflow for Experiment Tracking

**Status:** Accepted  
**Date:** 2024-11-14

**Context:**  
Need to track model experiments, compare metrics, and reproduce results. Options considered: MLflow, Weights & Biases, Neptune.

**Decision:**  
Use MLflow with local file-based tracking (`./mlruns`). It is open-source, requires no cloud account, and integrates natively with scikit-learn, XGBoost, and LightGBM.

**Consequences:**  
- Zero-cost, no account required, works offline.
- Local tracking store is git-ignored to keep the repo clean.
- UI available via `mlflow ui` for visual comparison.

---

## ADR-005: FastAPI Backend + Streamlit Dashboard

**Status:** Accepted  
**Date:** 2024-11-14

**Context:**  
The platform needs both a programmatic API (for integration) and a visual dashboard (for demonstration). Combining both in one framework would create tight coupling.

**Decision:**  
Use FastAPI as the backend API server and Streamlit as the standalone dashboard. The dashboard calls the API, maintaining separation of concerns.

**Consequences:**  
- API can be used independently by external systems.
- Dashboard is decoupled and can be replaced or extended.
- Two processes to run, but Docker Compose handles orchestration.

---

## ADR-006: Demand Forecasting Architecture (Module 1)

**Status:** Accepted  
**Date:** 2026-11-14

**Context:**  
Demand forecasting in supply chains requires both simple, resilient statistical benchmarks (to handle cold-start and seasonal rhythms) and machine learning models capable of capturing non-linear feature interactions without lookahead bias.

**Decision:**  
1. Adopt a tri-model architecture: `SeasonalNaiveForecaster` (7-day/30-day baseline), `XGBDemandForecaster`, and `LGBMDemandForecaster`.
2. Enforce strict temporal separation in feature engineering (`DemandFeaturePipeline`): lag offsets $\ge 1$ and rolling window calculations applied exclusively to target values shifted by 1 period.
3. Enforce the physical inventory invariant of non-negative demand ($y \ge 0$) at inference across all forecasters.
4. Complement MAE and RMSE with WAPE (Weighted Absolute Percentage Error) as the primary supply chain metric, avoiding division explosions on intermittent zero-demand periods.

**Consequences:**  
- Robust benchmark comparison against classical naive projection.
- Zero feature leakage into future horizon periods.
- High training throughput on multi-core CPU configurations without GPU requirements.
