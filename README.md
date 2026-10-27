# 🚀 AI-Driven Customer Churn Uplift Strategy

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Power BI](https://img.shields.io/badge/Power_BI-Desktop-F2C811?style=flat&logo=powerbi&logoColor=black)](https://powerbi.microsoft.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 💡 The Core Problem: Why Traditional Churn Models Fail

Most companies approach customer churn by predicting **"Who is most likely to cancel?"** and offering everyone discounts.

Here's why that approach wastes money:
- 💸 **Wasting budget on Sure Things:** Customers who were going to stay anyway still receive unneeded discounts.
- 🐻 **Poking the Sleeping Dogs:** Reaching out to annoyed, quiet customers often triggers immediate cancellations.
- 🕳️ **Chasing Lost Causes:** Spending retention resources on customers who are guaranteed to churn regardless of treatment.

### ✨ The Solution: Uplift Modeling (Causal AI)
Instead of predicting **risk**, Uplift Modeling predicts **persuadability** — identifying exactly which customers will change their behavior *only* if they receive an incentive.

## 🎯 The 4 Customer Retention Quadrants

| Quadrant | Customer Reaction | Business Action |
| :--- | :--- | :--- |
| 🟢 **Persuadables (The Target)** | Stay **only** if treated | 🎯 **Target aggressively with retention offers** |
| 🟡 **Sure Things** | Stay whether treated or not | 💰 **Do not contact** (Save marketing budget) |
| 🔴 **Lost Causes** | Leave regardless of treatment | 🛑 **Do not contact** (Zero ROI) |
| 🟠 **Sleeping Dogs** | Leave **only** if contacted | ⚠️ **Strictly do not disturb** (Let them be) |

## 📊 Key Highlights & Results

- **🎯 Precision Targeting:** Isolated **500+ high-potential churn targets** from a pool of 1,000+ customer profiles.
- **📉 Budget Optimization:** Reduced simulated marketing outreach waste by **30%** by eliminating spend on *Sure Things* and *Sleeping Dogs*.
- **📈 Interactive Power BI Dashboard:** Built dynamic DAX calculated measures and quadrant slicers plotting **Uplift Score vs. Baseline Retention Probability**.


https://github.com/user-attachments/assets/2c48237c-a5c0-4e10-973a-a5f8c9556c14

## 📈 Model Validation & MLOps Automation

Unlike standard classification models evaluated with ROC-AUC, Causal Uplift models must be evaluated on incremental treatment effect because a single customer cannot be observed in both Treated and Control states simultaneously.

* **Stratified Out-of-Sample Holdout:** Evaluated on a 20% holdout split stratified across treatment assignment and retention outcome ($T \times Y$).
* **Qini Coefficient & AUUC:** Computes the **Area Under the Uplift Curve (AUUC)** and normalized **Qini Coefficient** (`model_metrics.json`) to verify that ranking customers by predicted uplift outperforms random targeting.
* **Uplift @ Top 20%:** Measures net incremental retention gain in the top two deciles of targeted Persuadables.
* **Automated CI/CD Orchestration:** Scheduled via **GitHub Actions** (`daily_uplift.yml`) to execute daily at `00:00 UTC`, validate out-of-sample Qini metrics, and commit updated customer scores (`scored_customers.csv`) directly to the repository for Power BI ingestion.

## 🛠️ Tech Stack & Tools

- **Machine Learning & Modeling:** Python, Scikit-learn, Causal Inference / Uplift Modeling
- **Data Analysis & Manipulation:** Pandas, NumPy
- **Business Intelligence & Reporting:** Microsoft Power BI Desktop, DAX, Custom Quadrant Visuals
- **Version Control:** Git, GitHub

-

## 🚀 Quick Start Guide

### 1. Clone this repository

git clone [https://github.com/Satya12319/customer-churn-uplift.git](https://github.com/Satya12319/customer-churn-uplift.git)
cd customer-churn-uplift

### 2. Install dependencies
pip install -r requirements.txt

## 🤝 Connect with Me

* **GitHub:** [@Satya12319](https://github.com/Satya12319)
* **LinkedIn:** [Satya Prakash Singh](https://www.google.com/search?q=https://linkedin.com/in/satya-prakash-singh-266961318)
* **Email:** [ssatyasingh598@gmail.com](https://www.google.com/search?q=mailto%3Assatyasingh598%40gmail.com
