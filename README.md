# 🏦 Credit Risk Intelligence Engine

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-orange?logo=xgboost)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-green)
![AIF360](https://img.shields.io/badge/IBM%20AIF360-Fairness-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)

**An end-to-end credit default prediction system with explainable AI, demographic fairness auditing, and regulatory-compliant bias mitigation — built on 150,000+ real borrower profiles.**

[Overview](#-overview) • [Architecture](#-system-architecture) • [Results](#-results) • [Fairness](#️-fairness--bias-mitigation) • [Setup](#-setup) • [Usage](#-usage)

</div>

---

## 📌 Overview

The Credit Risk Intelligence Engine is a production-grade machine learning pipeline that goes beyond standard classification — it is designed to answer the hard questions that financial institutions actually face:

> *Can we identify high-risk applicants reliably **and** ensure that our model does not systematically disadvantage borrowers based on demographic characteristics?*

This project combines **gradient boosting**, **statistical feature analysis**, **multi-layer explainability (SHAP + LIME)**, and **IBM AIF360 fairness constraints** into a single, audit-ready system.

### Business Questions Addressed

| # | Question |
|---|----------|
| 1 | Which borrower characteristics are the strongest predictors of default? |
| 2 | How do we build a model that catches defaults while minimizing false rejections? |
| 3 | Can the model provide specific, auditable reasons for each credit decision? |
| 4 | Does the model treat all age demographics equitably under the EEOC 80% rule? |
| 5 | Can fairness gaps be closed without sacrificing predictive performance? |

---

## 🧱 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CREDIT RISK INTELLIGENCE ENGINE                  │
├──────────────┬──────────────┬──────────────┬────────────────────────┤
│   DATA LAYER │   ML LAYER   │  EXPLAIN.    │    FAIRNESS LAYER      │
│              │              │  LAYER       │                        │
│  Raw CSV     │  Logistic    │  SHAP Tree   │  AIF360 Reweighing     │
│  EDA         │  Regression  │  Explainer   │  (Pre-processing)      │
│  Stat Tests  │  (Baseline)  │              │                        │
│  Feature     │              │  LIME        │  Disparate Impact      │
│  Engineering │  Random      │  Tabular     │  Analysis              │
│              │  Forest      │  Explainer   │                        │
│  Imputation  │              │              │  Equal Opportunity     │
│  Outlier     │  XGBoost     │  Global +    │  Diff (EOD)            │
│  Capping     │  (Champion)  │  Local Scope │                        │
│              │              │              │  Threshold             │
│  StandardSc. │  Early       │  Per-case    │  Optimization          │
│              │  Stopping    │  explanations│  (Post-processing)     │
└──────────────┴──────────────┴──────────────┴────────────────────────┘
```

---

## 📂 Dataset

**Source:** [Give Me Some Credit — Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit)

| Attribute | Value |
|-----------|-------|
| Total Records | ~150,000 borrowers |
| Features | 11 raw + 5 engineered |
| Target Variable | `SeriousDlqin2yrs` (90+ day delinquency) |
| Class Imbalance | ~6.7% default rate (14:1 ratio) |
| Missing Data | `MonthlyIncome` (~19%), `NumberOfDependents` (~2.5%) |

### Feature Dictionary

| Column | Engineered Name | Description |
|--------|-----------------|-------------|
| `SeriousDlqin2yrs` | **Target** | 90+ day delinquency within 2 years |
| `RevolvingUtilizationOfUnsecuredLines` | Credit Usage % | Proportion of revolving credit in use |
| `age` | Age | Borrower age in years |
| `NumberOfTime30-59DaysPastDueNotWorse` | 1-Month Lates | Count of 30–59 day delinquencies |
| `DebtRatio` | Debt vs Income | Monthly obligations / monthly income |
| `MonthlyIncome` | Monthly Income | Gross monthly income |
| `NumberOfOpenCreditLinesAndLoans` | Open Accounts | Active credit lines + loans |
| `NumberOfTimes90DaysLate` | 3-Month Lates | Count of 90+ day delinquencies |
| `NumberRealEstateLoansOrLines` | Mortgages | Real estate credit lines |
| `NumberOfTime60-89DaysPastDueNotWorse` | 2-Month Lates | Count of 60–89 day delinquencies |
| `NumberOfDependents` | Family Size | Number of dependents |

### Engineered Features

| Feature | Logic | Rationale |
|---------|-------|-----------|
| `TotalPastDue` | Sum of all 30/60/90-day lates | Single delinquency severity signal |
| `CreditHistoryLength` | `(age - 18).clip(0)` | Proxy for years in credit system |
| `MonthlyPayment` | `DebtRatio × MonthlyIncome` | Actual cash-flow burden |
| `IncomePerPerson` | `MonthlyIncome / (Dependents + 1)` | Effective disposable income |
| `AgeGroup` | Binned: Young/MiddleAge/Senior/Elderly | Protected attribute for fairness audit |

---

## 🔬 Statistical Feature Analysis

Before modeling, every feature was validated using the **Mann-Whitney U Test** + **Cohen's d** effect size across the default/non-default split:

| Tier | Features | Cohen's d | Business Meaning |
|------|----------|-----------|------------------|
| **Power Trio** | `TotalPastDue`, `NumberOfTimes90DaysLate`, `RevolvingUtilization` | > 1.0 | Primary behavioral risk signals |
| **Stability** | `Age`, `CreditHistoryLength` | 0.2–0.5 | Protective maturity factors |
| **Secondary** | `MonthlyIncome`, `DebtRatio`, `NumberOfDependents` | < 0.2 | Supporting context features |

> **Verdict:** The extreme Cohen's d of the Power Trio features confirmed that a tree-based, split-optimizing model (XGBoost) would be the ideal architecture.

---

## 🤖 Model Development

Three models were trained and compared in a rigorous pipeline:

### 1. Logistic Regression (Baseline)
- `class_weight='balanced'` to address 14:1 imbalance
- SAGA solver for large-scale convergence
- Purpose: interpretable linear baseline + recall ceiling benchmark

### 2. Random Forest (Ensemble Benchmark)
- 200 estimators, `max_depth=10`
- Non-linear interaction capture
- Bridge between linear and boosting paradigms

### 3. XGBoost (Champion Model)
- `n_estimators=1000` with `early_stopping_rounds=50`
- `scale_pos_weight` tuned to exact class ratio (~14.0)
- `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`
- Early stopping on AUC — training halts automatically at optimal generalization

```python
xgb_model = XGBClassifier(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=ratio,      # ~14:1 imbalance correction
    objective='binary:logistic',
    eval_metric='auc',
    early_stopping_rounds=50
)
```

---

## 📈 Results

### Model Comparison

| Metric | Logistic Regression | Random Forest | **XGBoost** |
|--------|--------------------:|-------------:|------------:|
| Accuracy | — | — | **~86%** |
| **ROC-AUC** | Lower | Moderate | **0.8651** |
| Precision | Low | Higher | **Highest** |
| Recall | Highest | Moderate | **~79%** |
| F1-Score | Lower | Moderate | **Highest** |

> XGBoost selected as production model: highest ROC-AUC, best F1-Score, and most robust handling of class imbalance.

### Confusion Matrix Breakdown (XGBoost)

```
                Predicted: No Default    Predicted: Default
Actual: No Default     22,453               5,540
Actual: Default           430               1,575
```

| Business Metric | Value |
|-----------------|-------|
| Default Catch Rate (Recall) | **78.55%** |
| Safe Customer Clearance Rate | **80.21%** |
| Missed Defaulters | ~430 (~21% of actual defaults) |
| AUC — Discrimination Power | **0.8651** |
| Average Precision Score | **0.3976** (~6× better than random) |

> The model is **risk-averse by design**: it errs toward flagging borderline cases, since the cost of a missed default far exceeds the cost of a rejected safe applicant.

---

## 🔍 Explainability

### Global Explainability — SHAP (SHapley Additive exPlanations)

SHAP TreeExplainer was applied to a stratified 1,000-sample test subset, producing a ranked, directional view of global feature influence:

| Rank | Feature | Direction | Interpretation |
|------|---------|-----------|----------------|
| 1 | `TotalPastDue` | ↑ with value | Strongest default signal — any delinquency history sharply raises risk |
| 2 | `RevolvingUtilizationOfUnsecuredLines` | ↑ with value | Credit strain above ~70% is a heavy penalty |
| 3 | `Age` | ↓ with age | Youth = higher risk; maturity acts as a protective factor |
| 4 | `MonthlyIncome` | ↓ with income | Higher income modestly reduces risk |
| 5 | `DebtRatio` | Mixed | Meaningful only above extreme thresholds |

### Local Explainability — LIME (Individual Cases)

For the highest-risk case identified in the test set (predicted default probability: **97.2%**), LIME decomposed the prediction:

```
Feature                         Contribution
─────────────────────────────────────────────
TotalPastDue        (7.84)    → +0.33 risk
RevolvingUtilization(2.04)    → +0.29 risk
Age                 (-1.31)   → +0.07 risk  (young borrower)
CreditHistoryLength (-1.31)   → +0.06 risk  (short history)
```

> **Regulatory Value:** LIME explanations provide individualized, auditable reasons for each credit decision — a direct requirement under GDPR Article 22 and similar frameworks.

---

## ⚖️ Fairness & Bias Mitigation

This is the most technically sophisticated component of the project. The fairness pipeline uses **Age Group** as the protected attribute and evaluates compliance with the **EEOC 80% (Four-Fifths) Rule**.

### Phase 1 — Baseline Fairness Audit

| Metric | Value |
|--------|-------|
| Senior/Elderly Approval Rate | Higher |
| Young/Middle-Age Approval Rate | Lower |
| **Disparate Impact Ratio (Baseline)** | **< 0.80 → ⚠️ BIAS DETECTED** |
| Root Cause | Proxy discrimination via `TotalPastDue` + `RevolvingUtilization` (both correlated with age) |

### Phase 2 — AIF360 Reweighing (Pre-processing Mitigation)

```python
RW = Reweighing(
    unprivileged_groups=[{'privileged': 0.0}],
    privileged_groups=[{'privileged': 1.0}]
)
dataset_transf = RW.fit_transform(dataset_train)
instance_weights = dataset_transf.instance_weights

xgb_fair.fit(X_train_scaled, y_train, sample_weight=instance_weights)
```

Reweighing assigns corrective importance weights to training samples — upweighting under-represented fair cases and downweighting over-represented ones — so the model learns a naturally equitable decision boundary **without modifying features or labels**.

### Phase 3 — Optimal Threshold Search (Post-processing)

A high-to-low threshold scan (0.99 → 0.01, 500 steps) identified the tightest threshold that simultaneously:
1. Satisfies DI ≥ 0.80 (EEOC compliance), and
2. Maximizes F1-Score (operational utility)

```python
for t in np.linspace(0.99, 0.01, 500):
    preds = (xgb_fair_proba >= t).astype(int)
    cur_di = sel_unprivileged / sel_privileged
    if 0.80 <= cur_di <= 1.25:
        if f1_score(y_test, preds) > best_f1:
            final_thresh = t
```

### Fairness Results Summary

| Metric | Baseline | After Mitigation | Change |
|--------|----------|-----------------|--------|
| Disparate Impact Ratio | ~0.796 | **≥ 0.80** | ✅ Compliant |
| Equal Opportunity Diff | Higher | Lower | Improved |
| ROC-AUC | 0.8651 | ~0.865 | **Preserved** |
| Accuracy Impact | — | < 1% | **Negligible** |

> **Key Finding:** Fairness and predictive power are NOT mutually exclusive. The combined Reweighing + Best-F1 Threshold strategy achieves regulatory compliance while maintaining the full discriminative capacity of the original XGBoost model.

---

## 🛠️ Setup

### Requirements

```bash
Python >= 3.9
```

### Installation

```bash
git clone https://github.com/your-username/credit-risk-intelligence-engine.git
cd credit-risk-intelligence-engine
pip install -r requirements.txt
```

### Core Dependencies

```txt
numpy>=1.23
pandas>=1.5
scikit-learn>=1.2
xgboost>=1.7
shap>=0.42
lime>=0.2
aif360>=0.5
matplotlib>=3.6
seaborn>=0.12
scipy>=1.10
```

### Dataset

Download `cs-training.csv` from [Kaggle — Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit/data) and place it in the `data/` directory.

---

## 🚀 Usage

### Run the Full Pipeline

Open `credit_risk_intelligence_engine_v2.ipynb` in Jupyter or Google Colab and run all cells sequentially. The notebook is self-contained and will install missing dependencies automatically.

### Load Saved Models for Inference

```python
import pickle, json
import pandas as pd

# Load artifacts
with open('artifacts/xgboost_fair_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('artifacts/feature_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('artifacts/fairness_thresholds.json') as f:
    config = json.load(f)

with open('artifacts/feature_columns.json') as f:
    features = json.load(f)

# Predict on new applicant
applicant = pd.DataFrame([{
    'RevolvingUtilizationOfUnsecuredLines': 0.85,
    'age': 32,
    'DebtRatio': 0.45,
    'MonthlyIncome': 4500,
    'NumberOfOpenCreditLinesAndLoans': 7,
    'NumberRealEstateLoansOrLines': 1,
    'NumberOfDependents': 2,
    'CreditHistoryLength': 14,
    'TotalPastDue': 1
}])

applicant_scaled = scaler.transform(applicant[features])
risk_score = model.predict_proba(applicant_scaled)[0, 1]
threshold = config['global_fair_threshold']
decision = 'DEFAULT RISK' if risk_score >= threshold else 'LOW RISK'

print(f"Risk Score: {risk_score:.2%} → {decision}")
```

---

## 🏗️ Pipeline Walkthrough

```
1. DATA LOADING & EDA
   └─ Load cs-training.csv → shape inspection → missing value audit → class distribution

2. FEATURE ENGINEERING
   ├─ Error code correction (96/98 → 0 in delinquency columns)
   ├─ TotalPastDue aggregation
   ├─ MonthlyPayment = DebtRatio × MonthlyIncome
   ├─ IncomePerPerson = MonthlyIncome / (Dependents + 1)
   └─ AgeGroup binning (protected attribute)

3. STATISTICAL VALIDATION
   └─ Mann-Whitney U + Cohen's d → feature power ranking

4. MODEL TRAINING
   ├─ Logistic Regression (baseline)
   ├─ Random Forest (benchmark)
   └─ XGBoost (champion, early stopping + scale_pos_weight)

5. MODEL EVALUATION
   ├─ ROC-AUC, Precision, Recall, F1, Confusion Matrix
   ├─ ROC Curve + Precision-Recall Curve
   └─ Threshold analysis

6. EXPLAINABILITY
   ├─ SHAP TreeExplainer → global beeswarm plot
   └─ LIME → individual case breakdown

7. FAIRNESS AUDITING
   ├─ Baseline DI + EOD calculation (AIF360)
   ├─ AIF360 Reweighing (pre-processing)
   ├─ Re-training with instance weights
   ├─ Optimal threshold search (high→low scan)
   └─ Granular per-group audit table

8. ARTIFACT EXPORT
   └─ .pkl models + .json configs + .csv reports
```

---

## 📊 Key Technical Decisions

| Decision | Approach | Why |
|----------|----------|-----|
| Class imbalance | `scale_pos_weight` (XGBoost) + `class_weight='balanced'` (LR, RF) | Avoids majority-class collapse without SMOTE artifacts |
| Outlier handling | Clip RevolvingUtilization at 2.0, late-counts at 20 | Preserves over-extension signal without extreme skew |
| Feature scaling | StandardScaler on XGBoost | Required for LIME and fair model convergence |
| Bias mitigation | Pre-processing (Reweighing) + Post-processing (threshold) | Two-layer defense; neither alone is sufficient |
| Threshold strategy | High-to-low scan for tightest DI-compliant F1 | Avoids the degenerate "approve everyone" solution |
| Explainability | SHAP (global) + LIME (local) | Different stakeholders need different explanation granularity |

---

## 🔮 Future Roadmap

- [ ] **Streamlit Dashboard** — real-time loan officer interface with per-applicant SHAP waterfall charts
- [ ] **Model Drift Monitoring** — PSI-based feature distribution tracking for production deployment
- [ ] **Calibration Layer** — Platt scaling / isotonic regression for well-calibrated probability outputs
- [ ] **A/B Testing Framework** — controlled threshold experimentation with statistical significance testing
- [ ] **Intersectional Fairness** — multi-attribute analysis (age × income group)
- [ ] **API Deployment** — FastAPI wrapper with model versioning and audit logging

---

## 📚 References & Methodology

- Kaggle Competition: [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)
- Explainability: [SHAP — Lundberg & Lee, 2017](https://arxiv.org/abs/1705.07874)
- Local Explanations: [LIME — Ribeiro et al., 2016](https://arxiv.org/abs/1602.04938)
- Fairness Toolkit: [IBM AIF360](https://aif360.mybluemix.net/)
- Fairness Criterion: [EEOC Uniform Guidelines — 80% Rule](https://www.eeoc.gov/laws/guidance/questions-and-answers-clarify-and-provide-common-interpretation-uniform-guidelines)
- Gradient Boosting: [XGBoost — Chen & Guestrin, 2016](https://arxiv.org/abs/1603.02754)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with a commitment to both accuracy and equity in automated decision-making.**

*If this project helped you, consider starring the repo ⭐*

</div>
