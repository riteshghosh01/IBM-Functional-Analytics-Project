# 🚀 Decision360 — Business Intelligence & Decision Support Platform

**Decision360** is an end-to-end Business Intelligence and Decision Support Platform designed to transform raw business data into meaningful insights and actionable decisions.

The platform follows the complete analytics lifecycle:

> **Descriptive → Diagnostic → Predictive → Prescriptive Analytics**

Built entirely in **Python** with an interactive **Streamlit dashboard**, Decision360 enables businesses to understand what happened, identify why it happened, predict what may happen next, and determine what actions should be taken.

---

## 📌 Project Overview

Making business decisions based only on raw data can be difficult. Decision360 bridges this gap by combining data processing, analytics, machine learning, visualization, and business rules into a single decision-support platform.

The system allows users to upload their own **CSV or Excel datasets**, automatically identify relevant columns, perform analytics, and generate insights without requiring extensive manual configuration.

### Decision360 helps answer four key questions:

| Analytics Stage     | Business Question              |
| ------------------- | ------------------------------ |
| 📊 **Descriptive**  | What happened?                 |
| 🔍 **Diagnostic**   | Why did it happen?             |
| 🔮 **Predictive**   | What is likely to happen next? |
| 🎯 **Prescriptive** | What should we do about it?    |

---

# ✨ Key Features

## 📊 1. Executive Dashboard

Provides a high-level overview of business performance through dynamic KPIs and interactive visualizations.

**Features include:**

* Revenue and sales KPIs
* Period-over-period comparison
* Sales trend analysis
* Performance summaries
* Driver contribution analysis
* Business performance indicators

---

## 🔍 2. Diagnostic Analytics

Helps users understand the reasons behind changes in business performance.

The system analyzes major business drivers and identifies factors contributing to increases or decreases in performance.

**Example insights:**

* Why did sales decrease?
* Which product category contributed most to the change?
* Which region or customer segment is underperforming?
* What factors are driving revenue growth?

---

## 🔮 3. Predictive Forecasting

Decision360 uses machine learning to estimate future demand and business performance.

### Forecasting approach

* Linear Regression
* Time-based trend analysis
* Seasonality adjustment
* Historical demand patterns
* Model evaluation

The forecasting module can help businesses anticipate future demand and plan inventory, production, and sales strategies accordingly.

---

## 🚨 4. Anomaly Detection

The platform identifies unusual business activity using statistical analysis.

### Technique

**Z-Score Based Anomaly Detection**

Transactions or observations that significantly deviate from normal behavior are flagged for further investigation.

Potential applications include:

* Unusual sales spikes
* Unexpected revenue drops
* Abnormal transaction values
* Potential fraud indicators
* Operational irregularities

---

## 👥 5. Customer Intelligence

Decision360 uses **RFM (Recency, Frequency, Monetary) analysis** to segment customers based on their purchasing behavior.

### RFM Dimensions

* **Recency** — How recently did the customer purchase?
* **Frequency** — How often does the customer purchase?
* **Monetary** — How much does the customer spend?

Customers can then be grouped into meaningful segments such as:

* ⭐ High-value customers
* 💎 Loyal customers
* 🌱 Potential customers
* ⚠️ At-risk customers
* 💤 Inactive customers

This enables businesses to design more targeted customer strategies.

---

## 📦 6. Inventory Intelligence

The inventory module evaluates stock availability against expected demand and supplier lead times.

### Key metrics

* Current stock
* Average daily demand
* Days of stock remaining
* Lead time
* Stock coverage
* Potential stock-out risk

This helps identify products that may require replenishment before inventory reaches critical levels.

---

## 🎯 7. Decision Center

The Decision Center converts analytical findings into practical business recommendations.

It combines:

**Driver Analysis + KPIs + Thresholds + Business Rules**

to generate actionable recommendations.

### Example

> **Sales declining + inventory shortage detected → Prioritize replenishment for affected products.**

Instead of simply showing charts, Decision360 attempts to answer the more important question:

> **"What should the business do next?"**

---

## 🧪 8. What-If Simulator

The What-If Simulator allows users to test hypothetical business scenarios before implementing them in the real world.

Users can experiment with variables such as:

* Price changes
* Discount percentages
* Demand assumptions
* Sales scenarios

The system estimates the potential impact using a **constant-elasticity demand model**.

### Example

A business can evaluate:

> "What could happen to demand if we increase the price by 10%?"

or

> "How might sales change if we offer a 15% discount?"

This supports data-driven decision-making rather than relying entirely on assumptions.

---

# 🧠 Analytics Lifecycle

Decision360 connects multiple analytical stages into a single workflow:

```text
                RAW BUSINESS DATA
                       │
                       ▼
              ┌─────────────────┐
              │ Data Cleaning   │
              │ & Preprocessing │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Descriptive     │
              │ Analytics       │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Diagnostic      │
              │ Analytics       │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Predictive      │
              │ Analytics       │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Prescriptive    │
              │ Analytics       │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Decision Center │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ What-If         │
              │ Simulation      │
              └─────────────────┘
```

---

# 📁 Project Structure

```text
decision360/
│
├── app.py
├── analytics.py
├── business_insights.py
├── data_cleaning.py
├── data_generator.py
├── data_loader.py
├── eda.py
├── kpi_analysis.py
├── predictive_analysis.py
├── visualization.py
├── requirements.txt
└── README.md
```

### Module Description

| Module                   | Purpose                                       |
| ------------------------ | --------------------------------------------- |
| `app.py`                 | Streamlit dashboard and application interface |
| `analytics.py`           | Core analytics and machine learning logic     |
| `business_insights.py`   | Business rules and recommendation engine      |
| `data_cleaning.py`       | Data cleaning and preprocessing               |
| `data_generator.py`      | Synthetic business dataset generation         |
| `data_loader.py`         | CSV/Excel loading and column inference        |
| `eda.py`                 | Exploratory Data Analysis                     |
| `kpi_analysis.py`        | Dynamic KPI calculation                       |
| `predictive_analysis.py` | Forecasting and model evaluation              |
| `visualization.py`       | Interactive charts and visualizations         |
| `requirements.txt`       | Python dependencies                           |

---

# 🧩 Core Modules

| Module                     | Purpose                                 | Technique                       |
| -------------------------- | --------------------------------------- | ------------------------------- |
| **Executive Dashboard**    | KPI tracking and performance monitoring | KPI & driver analysis           |
| **Diagnostic Analytics**   | Identify performance drivers            | Contribution analysis           |
| **Customer Intelligence**  | Customer segmentation                   | RFM scoring                     |
| **Forecasting**            | Predict future demand                   | Linear Regression + Seasonality |
| **Anomaly Detection**      | Identify unusual observations           | Z-Score                         |
| **Inventory Intelligence** | Monitor stock availability              | Days-of-stock vs. Lead Time     |
| **Decision Center**        | Generate recommendations                | Rule-based decision engine      |
| **What-If Simulator**      | Test business scenarios                 | Constant Elasticity Model       |

---

# 📈 Interactive Visualizations

Decision360 provides interactive visualizations for exploring business performance.

The dashboard can include:

* 📊 Sales performance charts
* 📈 Revenue trend analysis
* 👥 Customer segment distributions
* 📦 Inventory coverage charts
* 🚨 Anomaly detection plots
* 🔮 Forecast vs. actual charts
* 🎯 KPI comparison charts
* 📉 Driver contribution analysis

Charts are designed to make analytical findings easier for both technical and non-technical users to understand.

---

# 🤖 Machine Learning & Analytics

Decision360 incorporates lightweight and explainable analytical techniques rather than relying on complex black-box models.

### Current techniques

**Forecasting**

```text
Historical Data
      ↓
Time-Based Features
      ↓
Trend Estimation
      ↓
Seasonality Adjustment
      ↓
Demand Forecast
```

**Anomaly Detection**

```text
Business Metric
      ↓
Mean & Standard Deviation
      ↓
Z-Score Calculation
      ↓
Threshold Evaluation
      ↓
Anomaly Flag
```

**Customer Segmentation**

```text
Transaction Data
      ↓
RFM Calculation
      ↓
Quantile Scoring
      ↓
Customer Segments
      ↓
Targeted Strategies
```

---

# 🔄 Dynamic Dataset Support

One of Decision360's major features is its ability to work with different business datasets.

Users can upload:

* `.csv`
* `.xlsx`
* `.xls`

The platform attempts to automatically identify relevant fields such as:

```text
Date
Customer
Product
Category
Sales
Quantity
Price
Discount
Inventory
Lead Time
```

This reduces the amount of manual configuration required when working with new datasets.

---

# 🛠️ Technology Stack

| Technology       | Purpose                   |
| ---------------- | ------------------------- |
| **Python**       | Core programming language |
| **Streamlit**    | Interactive web dashboard |
| **Pandas**       | Data manipulation         |
| **NumPy**        | Numerical computation     |
| **Scikit-learn** | Machine learning          |
| **Plotly**       | Interactive visualization |
| **CSV / Excel**  | Data input formats        |

---

# 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone <repository-url>
cd decision360
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Generate Sample Data

```bash
python data_generator.py
```

## 4. Launch the Application

```bash
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

# 💡 Example Use Cases

Decision360 can support a wide range of business scenarios.

### Sales Analytics

Monitor sales performance, identify growth drivers, and investigate declining revenue.

### Demand Forecasting

Estimate future demand to support inventory and operational planning.

### Customer Strategy

Identify valuable, loyal, at-risk, and inactive customer segments.

### Inventory Optimization

Detect potential stock-out situations using stock coverage and lead-time analysis.

### Pricing & Promotions

Evaluate potential price and discount strategies using What-If simulations.

### Anomaly Detection

Identify unusual transactions or business activity that requires further investigation.

### Executive Reporting

Provide decision-makers with a centralized dashboard containing KPIs, trends, insights, forecasts, and recommendations.

---

# 🏗️ Architecture Highlights

## Modular Architecture

The analytical logic is separated from the Streamlit interface.

This means the core analytics can potentially be reused with another frontend or API layer in the future.

```text
                 ┌───────────────┐
                 │   Streamlit   │
                 │       UI      │
                 └───────┬───────┘
                         │
                         ▼
              ┌────────────────────┐
              │ Analytics Engine   │
              └─────────┬──────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   KPI Analysis    ML Models      Business Rules
        │               │               │
        └───────────────┼───────────────┘
                        ▼
                Decision Insights
```

---

## 🔍 Explainable Analytics

The platform focuses on transparent and understandable analytical methods.

Rather than simply producing predictions, Decision360 attempts to provide context around the results.

For example:

```text
Revenue ↓ 8.4%

Primary Drivers:
• Category A → -4.1%
• Category B → -2.7%
• Region X    → -1.6%

Recommended Action:
Review declining categories and investigate
inventory availability in Region X.
```

This makes the output more useful for business users and stakeholders.

---

# 🔄 End-to-End Decision Loop

Decision360 is designed around a continuous decision-support cycle:

```text
DATA
  ↓
CLEAN
  ↓
ANALYZE
  ↓
UNDERSTAND
  ↓
PREDICT
  ↓
RECOMMEND
  ↓
SIMULATE
  ↓
DECIDE
  ↓
MEASURE OUTCOME
  ↓
IMPROVE
```

The objective is not just to build another analytics dashboard, but to create a platform that connects **business data with business decisions**.

---

# 🔮 Future Enhancements

The platform can be extended with several advanced capabilities:

### ⚡ Real-Time Data

* Kafka integration
* WebSocket-based updates
* Streaming analytics

### 🧠 Advanced Machine Learning

* XGBoost
* Prophet
* LSTM
* Advanced time-series forecasting

### 🗄️ Data Warehouse

* MySQL/PostgreSQL integration
* Automated ETL pipelines
* Scheduled data refreshes

### 💬 Natural Language Analytics

A conversational interface that allows users to ask questions such as:

> "Why did revenue fall last month?"

> "Which customers are at risk?"

> "What products are likely to run out of stock?"

### 🔁 Closed-Loop Decision Tracking

Track:

```text
Recommendation
      ↓
Action Taken
      ↓
Expected Impact
      ↓
Actual Impact
      ↓
Performance Comparison
```

This would allow Decision360 to continuously learn from previous business decisions.

---

# 🎯 Project Objective

The primary objective of Decision360 is to provide a unified platform that moves beyond traditional reporting.

Instead of stopping at:

> **"Here is what happened."**

Decision360 aims to progress toward:

> **"Here is what happened, why it happened, what is likely to happen next, and what you can do about it."**

---

# 📌 Project Status

**Current Status:** 🚧 Active Development

### Implemented

* [x] Dynamic CSV/Excel data loading
* [x] Data cleaning and preprocessing
* [x] Executive KPI dashboard
* [x] Diagnostic analytics
* [x] RFM customer segmentation
* [x] Demand forecasting
* [x] Z-score anomaly detection
* [x] Inventory intelligence
* [x] Rule-based recommendations
* [x] What-If simulation
* [x] Interactive visualizations

### Planned

* [ ] Real-time data ingestion
* [ ] Advanced forecasting models
* [ ] Database integration
* [ ] Natural-language analytics
* [ ] Automated scheduled reporting
* [ ] Closed-loop decision measurement

---

# 👨‍💻 Built With

**Python • Streamlit • Pandas • NumPy • Scikit-learn • Plotly**

---

## ⭐ Why Decision360?

Decision360 brings multiple analytical capabilities together in one platform:

**Data → Insights → Prediction → Recommendation → Simulation → Decision**

The goal is to make business analytics more **accessible, explainable, interactive, and decision-oriented**.
