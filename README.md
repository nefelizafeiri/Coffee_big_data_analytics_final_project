# Coffee Shop Big Data Analytics — Final Project

A PySpark-based analytics project examining ~500K coffee shop transactions to improve customer experience, understand spending patterns, and build a rewards program targeting strategy.

---

## Project Objectives

1. **Wait Time** — Identify factors that drive customer wait time (regression).
2. **Purchase Amount** — Identify factors that drive customer expenditure (regression).
3. **Rewards Targeting** — Predict which non-members are most likely to join the rewards program (classification).

Each target is modeled with a baseline plus ensemble alternatives (Random Forest, Gradient Boosted Trees). Operational recommendations follow from the best-performing model.

---

## Dataset

The dataset contains one row per transaction across 13 columns:

| Column | Type | Description |
|---|---|---|
| `transaction_id` | string | Unique transaction identifier |
| `age` | int | Customer age |
| `income` | string | Income bracket (Under $25K → Over $100K) |
| `sex` | string | Customer sex |
| `rewards_member` | bool | Whether the customer is in the rewards program |
| `occupation` | string | Employed / Retired / Self Employed / Student |
| `num_items` | int | Number of items purchased |
| `purchase_method` | string | Cash / Credit Card / Mobile Payment |
| `wait_time` | double | Minutes waiting in line |
| `purchase_amount` | double | Total transaction amount (USD) |
| `store_location` | string | Downtown / Midtown / Uptown |
| `transaction_time` | int | Hour of day (6–23) |
| `day_of_week` | string | Day of the week |

> `coffee-Full.csv` is not included in the repository. The notebook auto-generates a synthetic dataset (502,313 rows) matching the original statistics on first run.

---

## Results Summary

| Target | Best Model | Key Metric | Top Driver |
|---|---|---|---|
| Wait time | GBT Regressor | RMSE ≈ 1.25 min | `rewards_member`, `transaction_time` |
| Purchase amount | GBT Regressor | R² ≈ 0.86 | `num_items`, `income` |
| Rewards membership | Random Forest | AUC ≈ 0.956 | `wait_time`, `num_items`, `purchase_amount` |

---

## Key Findings

- **Wait time** is driven by order size and hour of day — not demographics. Staff peak hours and offer a dedicated large-order lane to reduce bottlenecks.
- **Purchase amount** is mechanically linked to `num_items` (r = 0.85). Bundle offers and upsells at point of sale are the highest-leverage revenue lever.
- **Rewards targeting**: score all non-members with the classifier and market to the top decile — these are non-members whose transaction profile most closely resembles existing members.

---

## Methodology

1. **Data Loading & Cleaning** — Type casting, null checks, dropping index columns.
2. **EDA** — Descriptive stats, categorical frequencies, correlation matrix, conditional means, KMeans segmentation (4 clusters).
3. **Feature Engineering** — Ordinal encoding for income, StringIndexer + OneHotEncoder for nominals, VectorAssembler.
4. **Modeling** — 70/30 train/test split (seed=42). Three targets × three algorithms each.
5. **Cross-Validation** — 3-fold CV with hyperparameter grid on the best wait-time model.

---

## Technologies

- **PySpark 3.x / 4.x** — distributed processing, MLlib modeling
- **Python** — pandas, NumPy, Matplotlib
- **Jupyter Notebook**
- **Java 11+** (required by Spark)

---

## Getting Started

### Prerequisites

```bash
pip install pyspark findspark numpy pandas matplotlib jupyter
```

Java 11 or later must be installed. On Ubuntu/Debian:

```bash
sudo apt-get install openjdk-11-jdk
```

### Run

```bash
git clone https://github.com/nefelizafeiri/Coffee_big_data_analytics_final_project.git
cd Coffee_big_data_analytics_final_project
jupyter notebook Coffee_Final_Project.ipynb
```

Run all cells. If `coffee-Full.csv` is not present, the notebook generates a synthetic dataset automatically on the first cell run.

---

## Repository Structure

```
Coffee_big_data_analytics_final_project/
├── Coffee_Final_Project.ipynb   # Main analysis notebook
├── Coffee-Problem-Statement.pdf # Original project brief
└── README.md
```
