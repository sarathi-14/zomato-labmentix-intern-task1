# 🍽️ Zomato Customer Sentiment Prediction and Restaurant Analytics System

> **A comprehensive, production-grade Machine Learning & Natural Language Processing system for restaurant review sentiment classification and dining analytics.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Project Overview

This capstone project delivers an end-to-end Machine Learning (ML) and Natural Language Processing (NLP) system combining restaurant metadata with customer review text to predict multi-class review sentiment (**Positive**, **Neutral**, **Negative**) and uncover structural, culinary, and behavioral drivers of diner satisfaction on Zomato.

### Core Capabilities:
- **Data Ingestion & Integrity:** Validated Many-to-One Left Join linking 10,000 customer reviews with 105 restaurant metadata records.
- **Exploratory Data Analysis (EDA):** 15 in-depth visualizations adhering strictly to the Univariate, Bivariate, and Multivariate (UBM) framework.
- **Statistical Hypothesis Testing:** Kruskal-Wallis H-test, Pearson's Chi-Square Test of Independence, and Spearman's Rank-Order Correlation.
- **NLP Text Processing:** Contraction expansion, lowercasing, punctuation/URL/digit removal, tokenization, WordNet lemmatization, and sentiment-aware stopword filtering with **negation preservation**.
- **Cold-Start Generalization:** Group-aware train/test splitting (`GroupShuffleSplit` on `Restaurant`) preventing data leakage across establishments.
- **Model Benchmarking:** Multinomial Naive Bayes, Multi-Modal Logistic Regression, LinearSVC, and Random Forest.
- **Model Explainability:** Linear coefficient extraction uncovering key positive and negative lexical drivers.
- **Production Deployment:** Serialized joblib model pipeline and an interactive Streamlit web application.

---

## 📂 Project Structure

```
Zomato-Sentiment-Analysis/
│
├── data/
│   ├── raw/
│   │   ├── Zomato Restaurant names and Metadata.csv    # 105 rows × 6 cols
│   │   └── Zomato Restaurant reviews.csv               # 10,000 rows × 7 cols
│   │
│   └── processed/
│       └── zomato_merged.csv                           # 9,961 clean rows × 32 cols
│
├── notebooks/
│   └── Zomato_ML_Project.ipynb                        # Fully executed submission notebook
│
├── models/
│   ├── zomato_sentiment_model.joblib                  # Serialized best model pipeline (LR)
│   └── model_metadata.json                            # Model hyperparameters & evaluation metrics
│
├── outputs/
│   ├── visualizations/                                # 15 EDA charts + model evaluation plots
│   ├── evaluation/
│   │   └── model_comparison.csv                       # Benchmark evaluation scores
│   └── reports/
│       └── evaluation_report.md                       # Comprehensive evaluation summary
│
├── src/
│   ├── data_cleaning.py                               # Ingestion, Left Join, and sanitization
│   ├── preprocessing.py                               # NLP text pipeline (NLTK)
│   ├── feature_engineering.py                         # Tabular & text feature transformations
│   ├── train.py                                       # Training, tuning & evaluation pipeline
│   └── predict.py                                     # CLI & programmatic inference interface
│
├── app.py                                             # Streamlit interactive web application
├── requirements.txt                                   # Python project dependencies
└── README.md                                          # Project documentation
```

---

## 📊 Dataset Catalog & Target Mapping

### Ingestion Statistics:
- **Restaurant Metadata:** 105 establishments across Hyderabad.
- **Customer Reviews:** 10,000 individual review records.
- **Merged & Sanitized Dataset:** 9,961 valid records after deduplication and rating sanitization.

### Target Variable (`Sentiment`):
Derived from continuous customer ratings ($1.0$ to $5.0$ stars):

| Sentiment Class | Star Rating Range | Observations | Percentage | Class Weight Strategy |
| :--- | :---: | :---: | :---: | :---: |
| **Negative** | $1.0 \le \text{Rating} \le 2.0$ | 2,428 | 24.4% | Cost-sensitive balanced loss |
| **Neutral** | $2.0 < \text{Rating} < 4.0$ | 1,259 | 12.6% | Cost-sensitive balanced loss |
| **Positive** | $4.0 \le \text{Rating} \le 5.0$ | 6,274 | 63.0% | Standard weighting |

---

## 🧪 Statistical Hypothesis Testing Results

All tests conducted at a $5\%$ significance level ($\alpha = 0.05$):

| Hypothesis | Test Applied | Test Statistic | p-Value | Outcome & Conclusion |
| :--- | :--- | :---: | :---: | :--- |
| **H1: Cost vs Rating** | Kruskal-Wallis H-Test | $H = 112.42$ | $< 0.0001$ | **Reject $H_0$**. Customer ratings differ significantly across Budget, Mid-Range, and Premium outlets. |
| **H2: Cuisine vs Sentiment** | Chi-Square ($\chi^2$) Independence | $\chi^2 = 84.62$ | $< 0.0001$ | **Reject $H_0$**. Customer sentiment is significantly dependent on primary cuisine. |
| **H3: Review Length vs Rating** | Spearman Rank Correlation | $\rho = -0.188$ | $< 0.0001$ | **Reject $H_0$**. Statistically significant inverse correlation: dissatisfied diners write longer complaint logs. |

---

## 🏆 Model Evaluation & Benchmark Comparison

Models evaluated on held-out restaurants using strict restaurant-grouped validation (`GroupShuffleSplit`):

| Model Architecture | Accuracy | Balanced Accuracy | **Macro F1-Score** | Weighted F1-Score | Minority Class Recall (Neg / Neu) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes (Baseline)** | **0.804** | 0.633 | 0.621 | 0.757 | 0.582 / 0.354 |
| **Logistic Regression (Champion)** 🏆 | 0.782 | **0.729** | **0.717** | **0.793** | **0.728 / 0.512** |
| **Linear Support Vector Classifier** | 0.791 | 0.688 | 0.691 | 0.787 | 0.685 / 0.441 |
| **Random Forest (Multi-Modal)** | 0.772 | 0.713 | 0.701 | 0.779 | 0.704 / 0.485 |

### Champion Model Selection Rationale:
**Multi-Modal Logistic Regression with Balanced Class Weighting** was selected because:
1. It achieved the highest **Macro F1-Score (0.717)** and **Balanced Accuracy (0.729)**, demonstrating superior detection of dissatisfied diners without succumbing to positive class bias.
2. It outputs natural, calibrated probability distributions for all three sentiment classes, essential for confidence scoring in production.
3. It provides transparent linear interpretability without black-box complexity.

---

## 🔍 Model Explainability & Key Drivers

- **Top Positive Sentiment Drivers:** `'delicious'` ($+3.42$), `'friendly'` ($+2.85$), `'best'` ($+2.71$), `'perfection'` ($+2.48$), `'amazing'` ($+2.41$), `'must try'` ($+2.20$), `'awesome'` ($+2.15$).
- **Top Negative Sentiment Drivers:** `'worst'` ($-3.84$), `'terrible'` ($-3.21$), `'disappointing'` ($-3.10$), `'rude'` ($-2.95$), `'horrible'` ($-2.88$), `'not good'` ($-2.75$), `'waste'` ($-2.61$).
- **Tabular Influences:** Higher dining `Cost` and attached `Pictures` positively associate with customer satisfaction; extreme review `Word_Count` correlates with service grievances.

---

## 🚀 Quick Start Guide

### 1. Installation

```bash
git clone https://github.com/sarathi-14/zomato-labmentix-intern-task1.git
cd zomato-labmentix-intern-task1
pip install -r requirements.txt
```

### 2. Run the Streamlit Web Application

```bash
streamlit run app.py
```

### 3. Run Programmatic Predictions

```python
import sys
sys.path.insert(0, 'src')
from predict import predict_sentiment

result = predict_sentiment(
    review="The butter chicken and garlic naan were extraordinary! Attentive staff.",
    cost=900.0,
    cuisine="North Indian",
    collections="Top-Rated",
    pictures=2
)
print(result)
# Output: {'sentiment': 'Positive', 'confidence': {'Negative': 0.02, 'Neutral': 0.06, 'Positive': 0.92}}
```

### 4. Open the Jupyter Notebook

```bash
jupyter notebook notebooks/Zomato_ML_Project.ipynb
```

---

## 📈 Business Recommendations

1. **Automated Managerial Triage:** Route customer reviews predicted as **Negative** directly to restaurant managers within 1 hour for immediate service recovery.
2. **Visual Presentation & Photography:** Aesthetic dish plating stimulates diner photo uploads, which associate with a **+14%** higher positive sentiment rate.
3. **Operational Discipline in Budget & Delivery Segments:** Budget dining experiences experience elevated rating variance; standardizing food preparation protocols will curtail 1-star reviews.
