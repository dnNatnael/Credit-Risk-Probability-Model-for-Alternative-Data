# Credit-Risk-Probability-Model-for-Alternative-Data

# 📘 Credit Scoring Project

A fully documented credit-scoring model aligned with Basel II regulatory standards.  
This project covers end-to-end development: data preparation, proxy target creation, modeling, validation, governance, and documentation.

---

## 📑 Table of Contents
- [Credit-Risk-Probability-Model-for-Alternative-Data](#credit-risk-probability-model-for-alternative-data)
- [📘 Credit Scoring Project](#-credit-scoring-project)
  - [📑 Table of Contents](#-table-of-contents)
- [Project Overview](#project-overview)
- [Credit Scoring Business Understanding](#credit-scoring-business-understanding)
  - [1. Influence of Basel II on Model Interpretability and Documentation](#1-influence-of-basel-ii-on-model-interpretability-and-documentation)
    - [🧩 Key Basel II Requirements Influencing Modeling](#-key-basel-ii-requirements-influencing-modeling)
      - [✔ Transparency and Explainability](#-transparency-and-explainability)
      - [✔ Documentation and Auditability](#-documentation-and-auditability)
      - [✔ Ethical and Fair-Lending Requirements](#-ethical-and-fair-lending-requirements)
  - [2. Need for a Proxy Variable When No Direct Default Label Exists](#2-need-for-a-proxy-variable-when-no-direct-default-label-exists)
    - [🔍 Why We Must Create a Proxy](#-why-we-must-create-a-proxy)
    - [⚠ Business Risks of Using Proxy Defaults](#-business-risks-of-using-proxy-defaults)
      - [1️⃣ Misalignment With True Default Behavior](#1️⃣-misalignment-with-true-default-behavior)
      - [2️⃣ Bias Introduction](#2️⃣-bias-introduction)
      - [3️⃣ Regulatory Defensibility Issues](#3️⃣-regulatory-defensibility-issues)
      - [4️⃣ Impact on Portfolio Strategies](#4️⃣-impact-on-portfolio-strategies)
  - [3. Trade-offs Between Interpretable and Complex Models in a Regulated Environment](#3-trade-offs-between-interpretable-and-complex-models-in-a-regulated-environment)
    - [🔵 Interpretable Models (Logistic Regression + WoE)](#-interpretable-models-logistic-regression--woe)
    - [🔴 Complex Models (Gradient Boosting, XGBoost, Random Forests)](#-complex-models-gradient-boosting-xgboost-random-forests)
    - [⚖ The Real-World Compromise](#-the-real-world-compromise)
- [Diagrams](#diagrams)
  - [1. PD Modeling Lifecycle](#1-pd-modeling-lifecycle)

---

# Project Overview

The objective of this project is to build a **credit scoring model** that predicts the probability that a borrower may default.  
Because financial institutions operate under strict regulatory frameworks (Basel II/III), the model must be:

- Transparent  
- Interpretable  
- Fair and non-discriminatory  
- Well-documented  
- Validated and monitored  

This repository includes all components needed to demonstrate a regulatory-grade credit scoring pipeline.

---

# Credit Scoring Business Understanding

## 1. Influence of Basel II on Model Interpretability and Documentation

The **Basel II Accord** establishes standards for credit risk measurement under Internal Ratings–Based (IRB) approaches.  
Its emphasis on **risk governance** directly shapes how credit scoring models must be built and documented.

### 🧩 Key Basel II Requirements Influencing Modeling

#### ✔ Transparency and Explainability
Basel II requires that:
- Each variable must have a defensible relationship to default risk.
- Model behavior must be interpretable for regulators, auditors, and credit officers.
- No “black box” decision-making can be used for regulatory capital calculations.

#### ✔ Documentation and Auditability
Institutions must maintain:
- Full data lineage documentation  
- Justification for feature engineering (e.g., WoE binning, monotonicity)  
- Detailed modeling assumptions  
- Validation reports (KS, ROC, Gini, PSI, calibration)  
- Stress testing results  
- Model monitoring framework  

#### ✔ Ethical and Fair-Lending Requirements
Models must:
- Avoid hidden bias  
- Produce consistent decisions across customer groups  
- Be explainable and defensible  

**Conclusion:**  
Basel II strongly favors **logistic regression + WoE** or similarly interpretable approaches.

---

## 2. Need for a Proxy Variable When No Direct Default Label Exists

The dataset does **not** include a direct “default” column (e.g., `default_flag`).  
But supervised models *require* a target variable.

### 🔍 Why We Must Create a Proxy
To train a PD model, we need to define what “default” means.  
Possible proxy definitions include:
- 90+ days past due  
- 3+ consecutive missed payments  
- Account written off  
- Assigned to collections  

Without a proxy:
- We cannot train or validate the model  
- No risk segmentation is possible  
- The PD model cannot be operationalized  

### ⚠ Business Risks of Using Proxy Defaults

#### 1️⃣ Misalignment With True Default Behavior
If the proxy does not reflect actual defaults:
- Non-risky customers may be rejected  
- Risky customers may be approved (leading to financial loss)  

#### 2️⃣ Bias Introduction
Proxies may unintentionally reflect:
- Operational issues  
- Customer behavior not linked to credit risk  
- Socioeconomic artifacts  

This can introduce **fairness and compliance risks**.

#### 3️⃣ Regulatory Defensibility Issues
Regulators can challenge:
- Why the proxy definition was chosen  
- Whether it reflects industry standards  
- Its statistical robustness  

#### 4️⃣ Impact on Portfolio Strategies
A poor proxy can distort:
- PD estimation  
- Risk-based pricing  
- Capital requirements (RWA)  
- Write-off policies  

---

## 4. Proxy Target Variable Engineering Implementation

This project implements a **RFM-based proxy target variable** to identify high-risk customers from transactional behavior patterns.

### 📊 Methodology

The proxy target variable (`is_high_risk`) is created using the following approach:

1. **RFM Metrics Calculation**
   - **Recency (R)**: Number of days since the customer's most recent transaction
   - **Frequency (F)**: Total number of transactions per customer
   - **Monetary (M)**: Total transaction amount per customer

2. **Customer Segmentation**
   - Apply K-Means clustering (n_clusters=3) on scaled RFM features
   - Use StandardScaler for feature normalization
   - Random state=42 for reproducibility

3. **High-Risk Identification**
   - Analyze cluster centroids to identify least engaged segment
   - High-risk cluster characterized by:
     - High Recency (long time since last transaction)
     - Low Frequency
     - Low Monetary value
   - Create binary target: `is_high_risk = 1` for least engaged cluster, `0` otherwise

### 🚀 Usage

#### Command Line Interface

```bash
# Basic usage with default parameters
python scripts/create_proxy_target.py

# Custom input/output paths
python scripts/create_proxy_target.py \
    --input data/raw/data.csv \
    --output data/processed/data_with_target.csv \
    --n-clusters 3 \
    --random-state 42
```

#### Python API

```python
from src.data_processing import create_proxy_target_variable
import pandas as pd

# Load your transactional data
df = pd.read_csv('data/raw/data.csv')

# Create proxy target variable
df_with_target, rfm_summary, metadata = create_proxy_target_variable(
    df=df,
    customer_id_col='CustomerId',
    transaction_date_col='TransactionStartTime',  # or 'TransactionDate'
    transaction_amount_col='Amount',  # or 'TransactionAmount'
    n_clusters=3,
    random_state=42
)

# The target variable is now available
print(df_with_target['is_high_risk'].value_counts())
```

### 📁 Output Files

The script generates three output files:

1. **`data/processed/data_with_target.csv`**
   - Original transactional data with added `is_high_risk` column

2. **`data/processed/rfm_summary.csv`**
   - Customer-level RFM metrics and cluster assignments
   - Columns: `CustomerId`, `Recency`, `Frequency`, `Monetary`, `Cluster`, `is_high_risk`

3. **`data/processed/target_metadata.json`**
   - Metadata including:
     - High-risk cluster ID
     - Target variable distribution
     - RFM statistics
     - Cluster summary

### 📚 Examples

See `examples/proxy_target_engineering_example.py` for comprehensive examples demonstrating:
- Basic RFM calculation
- Customer segmentation
- Complete pipeline usage
- Handling different column name formats

### ⚙️ Implementation Details

The implementation is located in:
- **`src/data_processing.py`**: Core functions for RFM calculation and clustering
- **`create_proxy_target.py`**: Command-line interface script
- **`examples/proxy_target_engineering_example.py`**: Usage examples

Key functions:
- `calculate_rfm_metrics()`: Compute RFM metrics per customer
- `segment_customers_with_kmeans()`: Apply K-Means clustering
- `identify_high_risk_cluster()`: Identify least engaged cluster
- `create_proxy_target_variable()`: Complete end-to-end pipeline

---

## 3. Trade-offs Between Interpretable and Complex Models in a Regulated Environment

### 🔵 Interpretable Models (Logistic Regression + WoE)

**Advantages**
- Highly explainable (regulator-friendly)
- Clear monotonic relationships
- Easy to calibrate and validate
- Stable performance over time
- Low governance burden

**Limitations**
- May underperform on nonlinear data  
- Requires manual engineering  

---

### 🔴 Complex Models (Gradient Boosting, XGBoost, Random Forests)

**Advantages**
- Higher predictive power  
- Automatically capture interactions and nonlinearities  
- Useful for internal analytics and risk segmentation  

**Limitations**
- Low interpretability  
- Requires SHAP/LIME explanation layers  
- Harder to monitor  
- More difficult for regulators to approve  
- Higher risk of overfitting  

---

### ⚖ The Real-World Compromise
Banks typically use:
- **Interpretable models for production decisions**, AND  
- **Complex models internally for portfolio insights**  

This ensures compliance without sacrificing analytical power.

---

# Diagrams

## 1. PD Modeling Lifecycle
```mermaid
flowchart LR
    A[Data Collection] --> B[Data Cleaning & Preprocessing]
    B --> C[Feature Engineering incl. WoE]
    C --> D[Model Development]
    D --> E[Model Validation]
    E --> F[Implementation]
    F --> G[Monitoring & Governance]
    G --> A
```

```mermaid
flowchart TB
    A[Board of Directors] --> B[Risk Committee]
    B --> C[Credit Risk Management]
    C --> D[Model Development Team]
    D --> E[Model Validation Team]
    E --> F[Documentation & Reporting]
    F --> G[Regulatory Submission]
```