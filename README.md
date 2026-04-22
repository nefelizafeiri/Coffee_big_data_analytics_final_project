# Coffee Shop Big Data Analytics — Final Project

A PySpark-based analytics project that examines coffee-shop transaction records to improve customer experience, understand spending patterns, and build a targeting strategy for a customer-rewards program.

## Problem Statement

This project analyzes a coffee shop's transaction data with three core goals:

1. **Explore** the data to identify patterns and trends in customer behavior.
2. **Model** the factors that influence customer wait times and purchase amounts using regression analysis.
3. **Recommend** operational improvements and develop a strategy for targeting non-members for the rewards program.

## Dataset

One row per transaction, ~1.8M rows, 13 columns:

| Column | Description |
|---|---|
| `transaction_id` | Unique identifier for each transaction |
| `age` | Age of the customer |
| `income` | Income bracket of the customer |
| `sex` | Sex of the customer |
| `rewards_member` | Whether the customer is enrolled in the rewards program |
| `occupation` | Occupation of the customer |
| `num_items` | Number of items purchased |
| `purchase_method` | Payment method (cash, credit card, etc.) |
| `wait_time` | Time spent waiting before the transaction (minutes) |
| `purchase_amount` | Total transaction amount (USD) |
| `store_location` | Store location where the transaction took place |
| `transaction_time` | Hour of day the transaction occurred |
| `day_of_week` | Day of the week the transaction occurred |

## Project Objectives

- Perform **exploratory data analysis (EDA)** to uncover distributions, correlations and anomalies.
- Build **regression models** to identify the key drivers of:
  - Customer wait time
  - Purchase amount
- Build a **classification model** that scores non-members for rewards-program targeting.
- Produce **data visualizations** that communicate findings clearly.
- Provide **actionable recommendations** for operations, revenue and customer acquisition.

## Methodology

1. **Data Cleaning** — drop index and identifier columns, cast `rewards_member` to 0/1, check missingness.
2. **Exploratory Data Analysis** — numeric summaries, categorical frequencies, sampled distribution plots, correlation heatmap, wait-time and spend breakdowns by store/hour/demographic, KMeans segmentation.
3. **Feature Engineering** — ordinal encoding of income, `StringIndexer` + `OneHotEncoder` for nominals, shared `VectorAssembler` feature spaces.
4. **Regression Analysis** — Linear Regression, Random Forest Regressor, GBT Regressor for both `wait_time` and `purchase_amount`.
5. **Rewards Classifier** — Logistic Regression and Random Forest Classifier, ranked by AUC; top-decile scoring on non-members drives the targeting list.
6. **Cross-Validation** — 3-fold CV on the winning wait-time model as a sanity check.
7. **Findings & Recommendations** — operational, revenue and targeting actions tied back to model evidence.

## Technologies Used

- **PySpark 3.5** — DataFrames + MLlib (runs locally, portable to a cluster)
- **pandas / NumPy** — small post-aggregation analysis
- **Matplotlib / Seaborn** — visualizations
- **Jupyter Notebook**

## Repository Structure

```
Coffee_big_data_analytics_final_project/
├── Coffee_Final_Project.ipynb     # Full PySpark analysis notebook
├── Coffee-Problem-Statement.pdf   # Original project brief
└── README.md
```

## Getting Started

```bash
git clone https://github.com/nefelizafeiri/Coffee_big_data_analytics_final_project.git
cd Coffee_big_data_analytics_final_project

pip install pyspark==3.5.1 pandas numpy matplotlib seaborn jupyter

jupyter notebook Coffee_Final_Project.ipynb
```

Before running, set the `DATA_PATH` variable in the **Load Data** section to point at your local copy of `coffee-Full.csv`. The default in the notebook is `/Users/danielregalado/Desktop/coffee-Full.csv`.

Java 8/11/17 must be installed for PySpark to start a local `SparkSession`.

## Results at a Glance

| Target | Best Model | Headline Metric | Dominant Driver |
|---|---|---|---|
| Wait time       | see §5.4 table | RMSE (min) | `num_items`, `transaction_time` |
| Purchase amount | see §6 table   | RMSE (USD) | `num_items`, income, occupation |
| Rewards member  | see §7.3 table | AUC        | `purchase_amount`, `num_items`, store/time |

See §9 of the notebook for the full set of operational, revenue and rewards-targeting recommendations.
