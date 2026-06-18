# AN ENSEMBLE MACHINE LEARNING APPROACH TO SME CREDIT SCORING USING ALTERNATIVE DATA FOR LOAN DEFAULT PREDICTION IN KENYA

## Project Overview

Access to credit remains a major challenge for Small and Medium Enterprises (SMEs) in Kenya. Traditional credit scoring methods often rely on formal credit histories and financial records, excluding many SMEs that operate in informal or semi-formal environments.

This project develops an alternative credit scoring system that leverages digital financial behavior, business activity indicators, utility payment patterns, and mobile money transaction data to predict loan default risk. Multiple machine learning models were evaluated, compared, and combined to identify the most effective approach for SME credit risk assessment.

The resulting system generates default probabilities, credit scores, risk categories, and lending recommendations that can support financial institutions, fintech companies, SACCOs, and digital lenders in making data-driven credit decisions.

---

## Research Title

**An Ensemble Machine Learning Approach to SME Credit Scoring Using Alternative Data for Loan Default Prediction in Kenya**

---

## Problem Statement

Many Kenyan SMEs lack sufficient credit history to qualify for loans under traditional credit assessment frameworks. As a result:

* Creditworthy businesses may be denied financing.
* Financial institutions face increased uncertainty when lending.
* Informal and underserved businesses remain financially excluded.

This project investigates whether alternative data sources can be used to build an accurate and interpretable machine learning-based credit scoring system.

---

## Objectives

### Main Objective

To develop an ensemble machine learning credit scoring model that predicts SME loan default risk using alternative data sources.

### Specific Objectives

* Develop predictive models for SME loan default.
* Engineer meaningful financial and behavioral features from alternative data.
* Compare the performance of multiple machine learning algorithms.
* Generate an interpretable credit score ranging from 300–850.
* Create automated lending recommendations based on predicted risk.
* Develop a deployable credit scoring application.

---

## Dataset Description

The dataset contains **25,000 SME loan applications** with information derived from alternative financial and business indicators.

### Key Variables

#### Business Characteristics

* Sector
* Years in operation
* Number of employees

#### Loan Information

* Loan amount
* Loan term
* Interest rate

#### Mobile Money Activity

* Average monthly M-Pesa inflow
* Average monthly M-Pesa outflow
* Monthly transaction count
* Savings ratio

#### Utility Data

* Electricity payment delays
* Electricity bills
* Water bills

#### Digital Presence

* Social media activity
* Online advertising expenditure
* Data spending
* Airtime spending

#### Sales Performance

* POS monthly sales
* POS transaction volume

#### Target Variable

* Loan Default (0 = Non-default, 1 = Default)

---

## Feature Engineering

Several domain-specific features were developed to enhance predictive performance.

### Financial Features

* Cash Flow Net
* Cash Flow Ratio
* Loan-to-Sales Ratio
* Credit Pressure
* Expense Pressure

### Business Features

* Business Activity Score
* Revenue per Employee
* Business Maturity

### Digital Features

* Digital Engagement Score
* No Digital Activity Indicator

### Utility Risk Features

* Utility Risk Score
* High Utility Risk Indicator

---

## Data Preprocessing

The following preprocessing steps were performed:

* Missing value verification
* Duplicate checking
* Correlation analysis
* Multicollinearity assessment
* One-hot encoding of categorical variables
* Feature scaling using StandardScaler
* Train-test split (80:20)
* Class balancing using SMOTE

---

## Machine Learning Models Evaluated

The following models were trained and compared:

1. Logistic Regression
2. Random Forest Classifier
3. XGBoost Classifier
4. LightGBM Classifier
5. CatBoost Classifier
6. Neural Network (MLP Classifier)

---

## Model Performance Comparison

| Model               | Accuracy  | Precision | Recall    | F1 Score  | ROC-AUC   |
| ------------------- | --------- | --------- | --------- | --------- | --------- |
| Logistic Regression | 73.7%     | 47.2%     | 73.8%     | 57.6%     | 82.1%     |
| Random Forest       | 78.0%     | 54.4%     | 56.2%     | 55.3%     | 80.8%     |
| XGBoost             | 80.3%     | 61.5%     | 49.8%     | 55.0%     | 81.7%     |
| LightGBM            | 80.4%     | 61.8%     | 49.7%     | 55.1%     | 81.9%     |
| CatBoost            | **80.7%** | **63.6%** | **47.5%** | **54.4%** | **82.1%** |
| Neural Network      | 73.9%     | 46.8%     | 56.7%     | 51.3%     | 76.7%     |

---

## Champion Model

### CatBoost Classifier

CatBoost achieved the strongest overall performance and was selected as the final production model.

### Reasons for Selection

* Highest Accuracy
* Highest Precision
* Strong ROC-AUC
* Handles categorical information effectively
* Robust against overfitting
* Produces interpretable feature importance

---

## Feature Importance

The most influential variables in predicting loan default were:

1. Utility Risk Score
2. Loan Term
3. Number of Employees
4. Digital Engagement
5. M-Pesa Monthly Inflow
6. Transaction Count
7. Credit Pressure
8. Interest Rate
9. Cash Flow Net
10. Loan-to-Sales Ratio

---

## Credit Score Development

Predicted default probabilities were transformed into a standardized credit score ranging from **300 to 850**.

### Score Formula

Credit Score = 850 − (Default Probability × 550)

### Risk Categories

| Score Range | Risk Category  |
| ----------- | -------------- |
| 750–850     | Very Low Risk  |
| 650–749     | Low Risk       |
| 550–649     | Medium Risk    |
| 450–549     | High Risk      |
| 300–449     | Very High Risk |

---

## Lending Decision Framework

| Credit Score | Decision              |
| ------------ | --------------------- |
| ≥ 720        | Approve Premium Loan  |
| 640–719      | Approve Standard Loan |
| 560–639      | Conditional Approval  |
| < 560        | Reject Application    |

---

## Explainability

Model explainability was conducted using SHAP (SHapley Additive Explanations).

SHAP analysis helped identify:

* Features driving default predictions
* Feature impact direction
* Individual loan application explanations
* Global model behavior

This improves transparency and trust in lending decisions.

---

## Fairness and Bias Analysis

A fairness audit was conducted across SME sectors.

The analysis evaluated:

* Sector-level default rates
* Approval rates
* Potential approval disparities
* Bias gaps between sectors

Results indicated reasonable consistency across sectors, although businesses in informal sectors exhibited higher default rates and lower approval rates.

---

## Deployment

The final model was deployed as:

### Streamlit Application

Features:

* SME loan application input form
* Real-time default prediction
* Credit score generation
* Risk categorization
* Automated lending recommendation

### Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* CatBoost
* SHAP
* Streamlit
* Joblib
* Matplotlib

---

## Project Structure


---

## Future Improvements

Potential enhancements include:

* Integration with real M-Pesa transaction APIs
* Ensemble stacking architecture
* Deep learning-based credit scoring
* Time-series borrower monitoring
* Dynamic credit limit recommendations
* Cloud deployment using AWS or Azure
* Real-time scoring API for financial institutions


## License

This project is intended for educational, research, and portfolio purposes.
