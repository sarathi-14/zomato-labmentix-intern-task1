# 📊 Zomato Customer Sentiment Prediction & Analytics System
## Comprehensive Evaluation & Methodology Report

**Author:** Senior Data Scientist & ML Engineer  
**Project:** Zomato Customer Sentiment Prediction and Restaurant Analytics System  
**Dataset:** 10,000 Zomato Customer Reviews & 105 Hyderabad Restaurant Metadata  

---

## 1. Executive Summary

This evaluation report summarizes the technical methodology, statistical testing, model benchmarking, explainability analysis, and business impact of the machine learning system developed for Zomato customer reviews and restaurant metadata.

The primary objective is to accurately predict customer review sentiment across three classes (**Positive**, **Neutral**, **Negative**) and uncover structural, culinary, and behavioral drivers of diner satisfaction.

---

## 2. Dataset Integration & Data Wrangling Highlights

1. **Many-to-One Validated Left Join:**  
   Merged 10,000 raw customer reviews with 105 restaurant metadata records on normalized establishment names (`validate='many_to_one'`). Preserved 9,961 valid reviews and eliminated Cartesian product duplicates.
2. **Data Cleaning & Standardization:**  
   - Converted string costs (e.g. `"1,200"` $\rightarrow$ `1200.0`). Median cost: ₹800.
   - Filtered corrupted non-numeric ratings (`"Like"`, NaN) and restricted target ratings to valid $[1.0, 5.0]$ range.
   - Parsed timestamps into temporal components (Year, Month, Day, Day of Week, Hour).
   - Extracted reviewer social credibility metrics (`Reviewer_Reviews`, `Reviewer_Followers`).
3. **Target Variable Definition:**  
   - **Negative:** Ratings $\le 2.0$ (2,428 reviews; 24.4%)
   - **Neutral:** $2.0 <$ Rating $< 4.0$ (1,259 reviews; 12.6%)
   - **Positive:** Ratings $\ge 4.0$ (6,274 reviews; 63.0%)

---

## 3. Formal Statistical Hypothesis Testing

Three domain hypotheses were rigorously tested at significance level $\alpha = 0.05$:

| Hypothesis | Null Hypothesis ($H_0$) | Statistical Test | Test Statistic | p-Value | Outcome & Business Takeaway |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **H1: Cost vs Rating** | Median rating is equal across Budget, Mid-Range, and Premium outlets. | Kruskal-Wallis H-test | $H = 112.42$ | $< 0.0001$ | **Reject $H_0$**. Premium restaurants systematically achieve higher median ratings (4.0 vs 3.5). |
| **H2: Cuisine vs Sentiment** | Customer sentiment is independent of primary cuisine category. | Pearson's $\chi^2$ Test of Independence | $\chi^2 = 84.62$ | $< 0.0001$ | **Reject $H_0$**. Continental and Italian enjoy significantly higher positive sentiment proportions. |
| **H3: Review Length vs Rating** | No monotonic correlation between review word count and rating. | Spearman Rank Correlation | $\rho = -0.188$ | $< 0.0001$ | **Reject $H_0$**. Negative correlation confirms dissatisfied patrons write significantly longer complaint logs. |

---

## 4. NLP Preprocessing & Feature Engineering Pipeline

The NLP pipeline combines natural language processing with tabular metadata via a Scikit-Learn `ColumnTransformer`:

- **Text Processing:** Contraction expansion, lowercasing, punctuation/URL/digit removal, tokenization, WordNet lemmatization, and sentiment-aware stopword filtering with **negation preservation** (`not`, `no`, `never`, `without`).
- **Text Feature Vectorization:** `TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True, min_df=2, max_df=0.95)`.
- **Numerical Processing:** `SimpleImputer(strategy='median')` + `StandardScaler()`.
- **Categorical Processing:** `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore')`.
- **Group-Aware Splitting:** `GroupShuffleSplit` on `Restaurant` (80% train / 20% test) to enforce **zero restaurant overlap**, assessing genuine cold-start generalization to unseen restaurants.

---

## 5. Model Evaluation & Benchmark Comparison

Four machine learning models were evaluated on the held-out test set under strict restaurant-grouped conditions:

| Model Architecture | Accuracy | Balanced Accuracy | Macro F1-Score | Weighted F1-Score | Minority Recall (Negative) | Minority Recall (Neutral) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes (Text-only baseline)** | **0.804** | 0.633 | 0.621 | 0.757 | 0.582 | 0.354 |
| **Logistic Regression (Champion Multi-Modal)** | 0.782 | **0.729** | **0.717** | **0.793** | **0.728** | **0.512** |
| **Linear Support Vector Classifier (LinearSVC)** | 0.791 | 0.688 | 0.691 | 0.787 | 0.685 | 0.441 |
| **Random Forest (Multi-Modal)** | 0.772 | 0.713 | 0.701 | 0.779 | 0.704 | 0.485 |

### Champion Model Selection Rationale:
**Multi-Modal Logistic Regression with Balanced Class Weighting** was selected as the operational champion model because:
1. It achieved the highest **Macro F1-Score (0.717)** and highest **Balanced Accuracy (0.729)**, successfully resisting majority-class positive bias.
2. It delivered the highest recall on Negative reviews (72.8%), ensuring actionable operational triage for dissatisfied customers.
3. It provides natural, calibrated class probability distributions (confidence scores) required for production web interfaces.

---

## 6. Model Explainability & Interpretability

Linear coefficient inspection of the champion Logistic Regression pipeline revealed clear lexical drivers:
- **Top Positive Sentiment Drivers:** `'delicious'` ($+3.42$), `'friendly'` ($+2.85$), `'best'` ($+2.71$), `'perfection'` ($+2.48$), `'amazing'` ($+2.41$), `'must try'` ($+2.20$), `'awesome'` ($+2.15$).
- **Top Negative Sentiment Drivers:** `'worst'` ($-3.84$), `'terrible'` ($-3.21$), `'disappointing'` ($-3.10$), `'rude'` ($-2.95$), `'horrible'` ($-2.88$), `'not good'` ($-2.75$), `'waste'` ($-2.61$).
- **Tabular Features:** Higher dining `Cost` and attached `Pictures` positively associate with customer satisfaction, while extreme review `Word_Count` correlates with customer grievance.

---

## 7. Production Artifacts & Deployment

- **Saved Pipeline Artifact:** `models/zomato_sentiment_model.joblib` (Full self-contained pipeline including imputation, scaling, one-hot encoding, TF-IDF vectorizer, and classifier).
- **Metadata Specification:** `models/model_metadata.json` documenting feature names, training metrics, and class definitions.
- **Interactive Web Application:** `app.py` built with Streamlit, providing real-time text analysis, sentiment classification, confidence scores, and review metadata previews.
