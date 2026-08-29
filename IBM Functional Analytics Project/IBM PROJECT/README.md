# Decision360 — Business Intelligence & Decision Support Platform

End-to-end analytics platform: descriptive → diagnostic → predictive →
prescriptive. Built in pure Python (pandas, scikit-learn, Streamlit, Plotly).

## Quick Start

```bash
pip install -r requirements.txt
python data_generator.py   # generates orders.csv and inventory.csv
streamlit run app.py
```

Streamlit will open the app in your browser (usually `http://localhost:8501`).

## Project Structure

```
decision360/
├── data_generator.py   # synthetic dataset with realistic patterns
├── analytics.py        # all analytics/ML logic (no UI dependency)
├── app.py               # Streamlit dashboard UI
├── requirements.txt
├── orders.csv            # generated on first run
├── inventory.csv         # generated on first run
└── README.md
```

## What Each Module Does

| Module | File function | Technique |
|---|---|---|
| Executive Dashboard | `compute_kpis`, `explain_kpi_change` | Period-over-period comparison + driver breakdown |
| Sales Analytics | (in `app.py`) | Grouped aggregation with interactive filters |
| Customer Intelligence | `rfm_segments` | RFM (Recency/Frequency/Monetary) segmentation via quantile scoring |
| Inventory Intelligence | (in `app.py`, uses `generate_inventory`) | Days-of-stock vs lead-time comparison |
| Forecasting | `forecast_metric` | Linear regression on trend + weekday seasonality adjustment |
| Anomaly Detection | `detect_anomalies` | Z-score threshold per dimension per day |
| Decision Center | `generate_recommendations` | Rule-based logic combining driver analysis + inventory status |
| What-If Simulator | `simulate_price_change`, `simulate_discount_sweep` | Constant-elasticity demand model |

## Why These Technical Choices (for your report)

- **Streamlit instead of React+FastAPI**: allows the full pipeline (data → analytics → ML →
  decision engine → UI) to live in one Python codebase, which is appropriate for a
  capstone timeline. The architecture in `analytics.py` is UI-agnostic, so it could be
  wrapped in a FastAPI + React frontend later without changing the underlying logic —
  worth stating explicitly as a "productionization path" in your report.
- **SQLite/CSV instead of MySQL**: the data layer is a plain pandas DataFrame loaded from CSV.
  If your rubric requires a relational database, swapping `pd.read_csv` for
  `pd.read_sql` against a MySQL connection is a small, mechanical change — the schema
  (`orders.csv` columns) is already relational-shaped (one row per order, foreign-key-like
  columns for region/product/customer).
- **Linear regression forecast instead of Prophet/ARIMA/LSTM**: chosen deliberately for
  explainability — you can show the exact coefficients driving the forecast, which is
  easier to defend in a viva than a black-box model, and avoids a dependency
  (`prophet`) that can be slow/unreliable to install under time pressure.
- **Rule-based recommendation engine instead of "AI decides"**: driver analysis finds
  *what* changed and by how much; a set of explicit if-then thresholds turns that into
  *actions*. This is intentionally transparent — say in your report that you chose
  interpretable rules over a black-box classifier because business recommendations
  need to be auditable.

## Suggested Report / Demo Narrative

1. Show the Executive Dashboard — "here's what's happening."
2. Click into the revenue driver breakdown — "here's why."
3. Go to Forecasting — "here's what's likely to happen next."
4. Go to Decision Center — "here's what the system recommends."
5. Go to What-If Simulator — "here's how we can test that recommendation before committing."

This is the same closed-loop story described in the original design brief (data → insight →
prediction → decision → simulation) and is more convincing live than describing each
module in isolation.

## Future Work (mention in report, don't build under time pressure)

- Real-time ingestion pipeline (Kafka/websockets) instead of static CSV
- Natural-language "AI Business Analyst" chatbot over the dataset
- MySQL-backed data warehouse with scheduled ETL
- More advanced forecasting (Prophet/XGBoost) with backtesting
- Closed-loop measurement: compare recommended action's expected impact vs actual outcome
