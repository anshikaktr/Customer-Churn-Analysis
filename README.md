# Customer Churn Analysis

An analysis of customer churn for a telecom company: data cleaning, exploratory data analysis (EDA), and classification models to identify customers at risk of leaving.

**Kaggle notebook:** https://www.kaggle.com/code/anshikakataria/customer-churn

---

## Problem Statement

Acquiring a new customer costs more than retaining an existing one. This project asks:

1. Which customer characteristics are most associated with churn?
2. Can we predict which customers are at risk of leaving?

## Dataset

- **Source:** Telco Customer Churn dataset on Kaggle
- **Size:** 7,043 rows x 21 columns
- **Target variable:** `Churn` (Yes/No)
- **Feature groups:** customer demographics, subscribed services, account/contract details, billing charges
- **Class balance:** 73.46% retained / 26.54% churned (moderately imbalanced)

## Tools

Python, Pandas, NumPy, Matplotlib/Seaborn, Scikit-learn, imbalanced-learn, Jupyter/Kaggle Notebooks

## Workflow

### 1. Data Cleaning
- **TotalCharges:** a small number of records had missing values. They made up only 0.156% of the dataset (11 rows), so they were dropped rather than imputed.
- **Tenure:** customers were first grouped into 12-month tenure bins (1-12, 13-24, 25-36, 37-48, 49-60 and 61-72 months) as a `tenure_group` feature, and the original `tenure` column was then removed.
- **customerID:** removed (identifier with no predictive value).
- **Target encoding:** `Churn` converted to binary (Yes = 1, No = 0).
- **Categorical variables:** converted into dummy variables.

### 2. Exploratory Data Analysis

**Charges and tenure**
- Churn is higher among customers with higher MonthlyCharges.
- Churn also appears higher at lower TotalCharges. On its own this looks odd, but it makes sense when the three variables are read together: customers with high monthly charges and short tenure have low total charges, and this group shows the highest churn. Low TotalCharges largely reflects short tenure.

**Higher churn associated with**
- Month-to-month contracts
- No online security and no tech support
- First year of subscription
- Fiber optic internet service
- Electronic check as payment method

**Lower churn associated with**
- Long-term contracts
- Customers with no internet service
- Tenure of 5+ years

**Little or no visible effect on churn**
- Gender, phone service availability, and number of multiple lines (also supported by the correlation heatmap)

> These are associations found in the data, not proven causes. For example, month-to-month customers may churn more because they have no contract lock-in, but the data alone cannot confirm that.

### 3. Model Building
- **Split:** 80/20 train/test (test set of 1,407 rows: 986 retained, 421 churned).
- **Models:** Decision Tree and Random Forest, each trained on the original data and on data resampled with **SMOTEENN** (SMOTE oversampling combined with Edited Nearest Neighbours cleaning).
- **Random Forest + SMOTEENN settings:** `max_depth=6`, `min_samples_leaf=5`, `random_state=100`.
- **Evaluation focus:** accuracy alone is misleading on imbalanced data, so precision, recall and F1 for the churn class (class 1) are reported.
- **Saved model:** Random Forest + SMOTEENN, saved as `model.sav`.

### 4. Results

All metrics except accuracy are for the churn class (class 1).

| Model | Test set | Accuracy | Precision | Recall | F1 |
|-------|----------|----------|-----------|--------|----|
| Decision Tree | Original (1,407 rows) | 0.78 | 0.68 | 0.51 | 0.58 |
| Random Forest | Original (1,407 rows) | 0.77 | 0.70 | 0.41 | 0.52 |
| Decision Tree + SMOTEENN | Resampled (1,172 rows) | 0.93 | 0.92 | 0.96 | 0.94 |
| Random Forest + SMOTEENN | Resampled (1,177 rows) | 0.94 | 0.93 | 0.97 | 0.95 |

**How to read this table**
- On the original data, both models catch fewer than about half of the churners (recall 0.41 to 0.51). The Decision Tree had higher recall than the Random Forest; the Random Forest had slightly higher precision.
- SMOTEENN raised every metric sharply, but those scores are **not directly comparable** with the first two rows. They were measured on resampled test sets where churners make up roughly 56-57% of rows, compared with about 27% in real data.
- Resampling was applied to the full dataset before the train/test split, so synthetic samples built from the same customers can appear in both the training and test sets. The SMOTEENN scores are therefore likely optimistic.

## Limitations

- The SMOTEENN models have not been evaluated on an untouched, original-distribution test set, so their real-world performance is unknown.
- No hyperparameter tuning or cross-validation was performed.
- The dataset is a single snapshot, so it cannot show how churn changes over time.
- The model was trained and saved but not tested outside the notebook.

## Project Status

**Completed:** data cleaning, EDA, model building and evaluation, model saved to `model.sav`.

**Not completed (out of scope for this version):**
- Flask web app / API to serve predictions from the saved model
- Tableau dashboard
- Re-evaluation of SMOTEENN with resampling applied to the training set only

This repository covers the analysis and modelling stages of the project.

## How to Run

```bash
git clone https://github.com/anshikaktr/Customer-Churn-Analysis.git
cd Customer-Churn-Analysis
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
jupyter notebook   # open the notebook and run all cells
```

## Author

Anshika Kataria
