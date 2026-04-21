Coffee Shop Big Data Analytics — Final Project
A data analytics project that examines coffee shop transaction records to improve customer experience, understand spending patterns, and build a targeting strategy for a customer rewards program.

Problem Statement
This project analyzes a coffee shop's transaction data with three core goals:

Explore the data to identify patterns and trends in customer behavior.
Model the factors that influence customer wait times and purchase amounts using regression analysis.
Recommend operational improvements and develop strategies to target customers for the rewards program based on their spending habits and preferences.
Dataset
The dataset contains one row per transaction, with the following fields:

Column	Description
transaction_id	Unique identifier for each transaction
age	Age of the customer
income	Income of the customer
sex	Sex of the customer
rewards_member	Whether the customer is enrolled in the rewards program
occupation	Occupation of the customer
num_items	Number of items purchased
purchase_method	Payment method (e.g., cash, credit card)
wait_time	Time spent waiting in line before the transaction (minutes)
purchase_amount	Total transaction amount (USD)
store_location	Store location where the transaction took place
transaction_time	Hour of day when the transaction occurred
day_of_week	Day of the week when the transaction occurred
Project Objectives
Perform exploratory data analysis (EDA) to uncover distributions, correlations, and anomalies.
Build regression models to identify the key drivers of:
Customer wait time
Purchase amount
Produce data visualizations that communicate findings clearly.
Provide actionable recommendations for:
Improving store operations and reducing wait times
Enhancing the customer experience
Targeting the right customers for the rewards program
Methodology
Data Cleaning — Handle missing values, outliers, and data type consistency.
Exploratory Data Analysis — Summarize distributions and relationships across demographic, behavioral, and temporal variables.
Feature Engineering — Encode categorical variables and prepare features for modeling.
Regression Analysis — Fit and evaluate models predicting wait time and purchase amount.
Segmentation & Targeting — Identify customer profiles most likely to benefit from or engage with the rewards program.
Recommendations — Synthesize insights into concrete operational and marketing actions.
Technologies Used
Python (pandas, NumPy, scikit-learn, statsmodels)
Data Visualization (Matplotlib, Seaborn)
Jupyter Notebook
Repository Structure
Coffee_big_data_analytics_final_project/
├── Coffee-Problem-Statement.pdf   # Original project brief
└── README.md
Getting Started
Clone the repository:
git clone https://github.com/nefelizafeiri/Coffee_big_data_analytics_final_project.git
cd Coffee_big_data_analytics_final_project
Install dependencies:
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn jupyter
Launch Jupyter Notebook and open the analysis notebook:
jupyter notebook
The stop hook will keep showing this until the commit is pushed — there's nothing I can do to silence it from my side.

Go to your repository on GitHub → click README.md → click the pencil icon to edit → paste the markdown I gave you → commit directly on main. That bypasses the git push entirely.

but in a markdown file or just to copy...

Just copy the text I gave — no file needed. Go to:

https://github.com/nefelizafeiri/Coffee_big_data_analytics_final_project/edit/main/README.md

Select all the existing text, paste the markdown I gave you, then click Commit changes.

Still the same unpushed commit. Did the GitHub edit work? If you committed via the web interface, let me check if it went through.

Used get file contents
The README on GitHub is still the old one-liner — the web edit hasn't been done yet.

Go to this URL, select all, paste, and commit:

https://github.com/nefelizafeiri/Coffee_big_data_analytics_final_project/edit/main/README.md

# Coffee_big_data_analytics_final_project it only has this!!! fix it without any mistakes

Used create or update file
I still can't write to GitHub from this session — both git and the API return 403. Here's exactly what to do in the GitHub web editor:

Go to https://github.com/nefelizafeiri/Coffee_big_data_analytics_final_project/edit/main/README.md
Press Ctrl+A to select everything
Press Delete
Paste the text below
Scroll down and click Commit changes
Copy everything between the lines:

# Coffee Shop Big Data Analytics — Final Project
A data analytics project that examines coffee shop transaction records to improve customer experience, understand spending patterns, and build a targeting strategy for a customer rewards program.
---
## Problem Statement
This project analyzes a coffee shop's transaction data with three core goals:
1. **Explore** the data to identify patterns and trends in customer behavior.
2. **Model** the factors that influence customer wait times and purchase amounts using regression analysis.
3. **Recommend** operational improvements and develop strategies to target customers for the rewards program based on their spending habits and preferences.
---
## Dataset
The dataset contains one row per transaction, with the following fields:
| Column | Description |
|---|---|
| `transaction_id` | Unique identifier for each transaction |
| `age` | Age of the customer |
| `income` | Income of the customer |
| `sex` | Sex of the customer |
| `rewards_member` | Whether the customer is enrolled in the rewards program |
| `occupation` | Occupation of the customer |
| `num_items` | Number of items purchased |
| `purchase_method` | Payment method (e.g., cash, credit card) |
| `wait_time` | Time spent waiting in line before the transaction (minutes) |
| `purchase_amount` | Total transaction amount (USD) |
| `store_location` | Store location where the transaction took place |
| `transaction_time` | Hour of day when the transaction occurred |
| `day_of_week` | Day of the week when the transaction occurred |
---
## Project Objectives
- Perform **exploratory data analysis (EDA)** to uncover distributions, correlations, and anomalies.
- Build **regression models** to identify the key drivers of:
  - Customer wait time
  - Purchase amount
- Produce **data visualizations** that communicate findings clearly.
- Provide **actionable recommendations** for:
  - Improving store operations and reducing wait times
  - Enhancing the customer experience
  - Targeting the right customers for the rewards program
---
## Methodology
1. **Data Cleaning** — Handle missing values, outliers, and data type consistency.
2. **Exploratory Data Analysis** — Summarize distributions and relationships across demographic, behavioral, and temporal variables.
3. **Feature Engineering** — Encode categorical variables and prepare features for modeling.
4. **Regression Analysis** — Fit and evaluate models predicting wait time and purchase amount.
5. **Segmentation & Targeting** — Identify customer profiles most likely to benefit from or engage with the rewards program.
6. **Recommendations** — Synthesize insights into concrete operational and marketing actions.
---
## Technologies Used
- **Python** (pandas, NumPy, scikit-learn, statsmodels)
- **Data Visualization** (Matplotlib, Seaborn)
- **Jupyter Notebook**
---
## Repository Structure
Coffee_big_data_analytics_final_project/
├── Coffee-Problem-Statement.pdf # Original project brief
└── README.md

---
## Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/nefelizafeiri/Coffee_big_data_analytics_final_project.git
   cd Coffee_big_data_analytics_final_project
Install dependencies:
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn jupyter
Launch Jupyter Notebook and open the analysis notebook:
jupyter notebook
