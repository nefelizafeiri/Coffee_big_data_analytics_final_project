# Coffee Shop Big Data Analytics — Final Project

A hybrid **PySpark + scikit-learn / XGBoost / Optuna / SHAP** analytics project that examines ~1.8M coffee-shop transactions to improve operations, understand spend drivers, and build a rewards-program targeting strategy.

## Problem Statement

1. **Explore** customer behavior patterns and trends.
2. **Model** the drivers of customer wait time and purchase amount (regression).
3. **Classify** rewards-program members to build a targeting strategy for non-members.

## Dataset

One row per transaction, ~1.8M rows, 13 columns:

| Column | Description |
|---|---|
| `transaction_id` | Unique identifier |
| `age` | Customer age |
| `income` | Income bracket |
| `sex` | Customer sex |
| `rewards_member` | Enrolled in rewards program (TRUE/FALSE) |
| `occupation` | Customer occupation |
| `num_items` | Items purchased |
| `purchase_method` | Payment method |
| `wait_time` | Wait time in minutes |
| `purchase_amount` | Transaction amount (USD) |
| `store_location` | Store |
| `transaction_time` | Hour of day |
| `day_of_week` | Day of the week |

## ML Workflow (best practices applied)

1. **Data cleaning** — PySpark ingest, null and duplicate checks, type casting.
2. **EDA** — numeric stats, categorical frequencies, distributions, correlation heatmap, wait-time / spend breakdowns, KMeans segmentation.
3. **Feature engineering** — ordinal `income`, one-hot nominals, derived `is_peak_hour` and `is_weekend`.
4. **Feature selection** — Variance Threshold + F-score + Mutual Information + RFE (Random Forest).
5. **Train / Validation / Test split** — 70/15/15, stratified for classification.
6. **Model zoo** — baseline Linear/Logistic Regression + tuned Random Forest + tuned GBM + tuned XGBoost.
7. **Hyperparameter search** — `GridSearchCV` (coarse) and **Optuna** TPE (fine, Bayesian) with early stopping.
8. **Final predictions** — refit on train+val, scored once on the held-out test set (honest generalization).
9. **SHAP explainability** — global (bar, beeswarm), local (waterfall), dependence plots.
10. **Recommendations** — operational, revenue, and targeting actions tied back to SHAP evidence.

## Technologies

- **PySpark 3.5** — big-data ingest, EDA, segmentation.
- **scikit-learn 1.4**, **XGBoost 2.0**, **Optuna 3.6**, **SHAP 0.45** — modeling and explainability.
- **pandas / NumPy / matplotlib / seaborn** — analysis and visualization.

## Repository Structure

```
Coffee_big_data_analytics_final_project/
├── Coffee_Final_Project.ipynb     # Full analysis notebook (EDA + ML + SHAP)
├── Coffee-Problem-Statement.pdf   # Original project brief
└── README.md
```

## Getting Started

```bash
git clone https://github.com/nefelizafeiri/Coffee_big_data_analytics_final_project.git
cd Coffee_big_data_analytics_final_project

pip install pyspark==3.5.1 pandas numpy matplotlib seaborn \
    scikit-learn==1.4.2 xgboost==2.0.3 optuna==3.6.1 shap==0.45.0 jupyter

jupyter notebook Coffee_Final_Project.ipynb
```

Set `DATA_PATH` in section 2 to the local location of `coffee-Full.csv` (default: `/Users/danielregalado/Desktop/coffee-Full.csv`). Java 8/11/17 is required for PySpark.

## Results at a Glance

| Target | Best Model | Test Metric | Top-2 SHAP Features |
|---|---|---|---|
| Wait time       | see §7.4 | RMSE (min) | `num_items`, `is_peak_hour` |
| Purchase amount | see §8.4 | RMSE (USD) | `num_items`, `income_ord`   |
| Rewards member  | see §9.3 | AUC        | `purchase_amount`, `num_items` |

See §10 of the notebook for the full set of operational, revenue and rewards-targeting recommendations, each grounded in SHAP evidence.
