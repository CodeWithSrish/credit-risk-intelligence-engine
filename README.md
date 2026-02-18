# 🏦 Credit Risk Intelligence Engine

> **Credit Risk Assessment with Fairness Constraints & Model Explainability**

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-orange)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-green)](https://shap.readthedocs.io/)
[![AIF360](https://img.shields.io/badge/IBM%20AIF360-Fairness-purple)](https://aif360.mybluemix.net/)
[![Kaggle](https://img.shields.io/badge/Dataset-Give%20Me%20Some%20Credit-20BEFF?logo=kaggle)](https://www.kaggle.com/c/GiveMeSomeCredit)

---

## 📖 Overview

This project builds an end-to-end **credit default prediction system** using the *Give Me Some Credit* Kaggle dataset. Beyond raw predictive performance, the notebook integrates a full **model explainability pipeline** (SHAP + LIME) and a **fairness auditing framework** (IBM AIF360) to ensure equitable lending decisions across demographic groups.

The system is designed to answer four core business questions:

1. Which customer characteristics are most strongly associated with credit default?
2. Can machine learning models reliably identify high-risk applicants before default occurs?
3. What are the primary drivers of default risk according to the model?
4. How should prediction thresholds be selected to balance risk detection and customer approval rates?

---

## 📂 Dataset

**Source:** [Give Me Some Credit — Kaggle Competition](https://www.kaggle.com/c/GiveMeSomeCredit)

The dataset contains financial and behavioural attributes of borrowers, where each row represents a credit applicant. The target variable is `SeriousDlqin2yrs` (renamed `Defaulted`) — whether the person experienced financial distress within two years.

| Original Column | Simple Name | Description |
|---|---|---|
| `SeriousDlqin2yrs` | Defaulted | Target: 1 if financial distress within 2 years |
| `RevolvingUtilizationOfUnsecuredLines` | Revolving Utilization | Credit card balance as % of credit limit |
| `age` | Age | Borrower's age in years |
| `NumberOfTime30-59DaysPastDueNotWorse` | Past Due 30–59 Days | Times 30–59 days late on payment |
| `DebtRatio` | Debt Ratio | Monthly debt payments / monthly gross income |
| `MonthlyIncome` | Monthly Income | Borrower's monthly income |
| `NumberOfOpenCreditLinesAndLoans` | Open Credit Lines | Number of open credit lines and loans |
| `NumberOfTimes90DaysLate` | Past Due 90+ Days | Times 90+ days late |
| `NumberRealEstateLoansOrLines` | Real Estate Loans | Number of real estate loans |
| `NumberOfTime60-89DaysPastDueNotWorse` | Past Due 60–89 Days | Times 60–89 days late |
| `NumberOfDependents` | Dependents | Number of dependents |

---

## 🔬 Project Structure

```
credit_risk_intelligence_engine.ipynb
│
├── 1. Business Questions & Project Scope
├── 2. Library Imports
├── 3. Data Loading & Exploration
├── 4. Data Preprocessing & Feature Engineering
├── 5. Exploratory Data Analysis (EDA)
├── 6. Statistical Testing for Key Risk Drivers
├── 7. Model Development
│   ├── Baseline: Logistic Regression
│   ├── Ensemble: Random Forest
│   └── Primary: XGBoost (Gradient Boosting)
├── 8. Model Evaluation
│   ├── Confusion Matrix
│   ├── ROC & Precision-Recall Curves
│   └── Model Comparison Summary
├── 9. Model Explainability
│   ├── SHAP (Global Feature Importance)
│   └── LIME (Individual Prediction Explanations)
├── 10. Fairness Auditing (IBM AIF360)
│   ├── Pre-Mitigation Bias Assessment
│   ├── Reject Option Classification (ROC) Mitigation
│   └── Post-Mitigation Fairness Audit
├── 11. Final Performance Summary
└── 12. Save Model & Artifacts
```

---

## ⚙️ Methodology

### Data Preprocessing
- Missing value imputation using median strategy (`SimpleImputer`)
- Feature scaling with `StandardScaler`
- Class imbalance handling via `scale_pos_weight` (XGBoost) and `class_weight='balanced'` (sklearn models)
- Engineered aggregate feature: **Total Past Due** (sum of all delinquency counts)

### Statistical Testing
Mann-Whitney U tests and KS tests are applied to confirm statistically significant differences (p < 0.05) between default and non-default groups across all key features. Cohen's D effect sizes are also reported to assess practical significance beyond statistical thresholds.

### Models Trained
| Model | Purpose |
|---|---|
| Logistic Regression | Interpretable baseline |
| Random Forest | Non-linear ensemble benchmark |
| **XGBoost** | Primary production model |

### Explainability
- **SHAP** — Global feature importance using TreeExplainer; summary and beeswarm plots to understand model behaviour across the full dataset.
- **LIME** — Local surrogate model for individual prediction explanations, showing per-feature contribution to a specific applicant's risk score.

### Fairness Auditing
- Protected attribute: **Age Group** (focus on 18–25 as unprivileged group)
- Metrics: Disparate Impact (80% Rule), Equal Opportunity Difference, Statistical Parity Difference
- Mitigation: **Reject Option Classification (ROC)** via IBM AIF360 — adjusts decision thresholds near the decision boundary to achieve demographic parity without retraining the model

---

## 📊 Key Results

| Metric | Before Mitigation | After Mitigation |
|---|---|---|
| Disparate Impact | 0.7964 ❌ | ~1.00 ✅ |
| Equal Opportunity Diff | Biased | Equalized ✅ |
| ROC-AUC | ~0.86 | Maintained |
| Recall (Defaults) | High | Maintained |

**Top Predictors (SHAP):**
1. Total Past Due (historical delinquency)
2. Revolving Utilization of Unsecured Lines
3. Age
4. Number of Times 90+ Days Late
5. Debt Ratio

---

## 💾 Output Artifacts

After running the full notebook, the following files are saved:

| File | Description |
|---|---|
| `xgboost_credit_model.pkl` | Trained XGBoost model |
| `feature_scaler.pkl` | Fitted StandardScaler |
| `fairness_thresholds.json` | Age-group-specific decision thresholds |
| `feature_columns.json` | List of feature names for inference |
| `model_performance_summary.csv` | Model comparison metrics |
| `fairness_metrics.csv` | Pre/post-mitigation fairness metrics |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost shap lime aif360
```

### Running the Notebook

1. Download the *Give Me Some Credit* dataset from [Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit) and place it in your working directory.
2. Open the notebook in Jupyter, JupyterLab, Google Colab, or Kaggle.
3. Run all cells sequentially from top to bottom.

> **Note:** The notebook was developed on Python 3.12 with Kaggle's hosted environment. LIME and AIF360 are installed inline via `!pip install` cells.

---

## 🎓 Future Improvements

1. **Production Integration** — Connect to a live credit underwriting pipeline
2. **Model Drift Monitoring** — Set up automated monitoring for performance and fairness metric degradation over time
3. **A/B Testing Framework** — Continuously evaluate threshold strategies and mitigation approaches on real-world outcomes

---

## 📚 References

- [IBM AIF360 Documentation](https://aif360.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [LIME Paper — Ribeiro et al. (2016)](https://arxiv.org/abs/1602.04938)
- [Give Me Some Credit — Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit)
- EEOC 80% (Four-Fifths) Rule for Disparate Impact Analysis


## 👩‍💻 Author

**Srishti Rajput**  
Credit Risk Intelligence Engine

---
 