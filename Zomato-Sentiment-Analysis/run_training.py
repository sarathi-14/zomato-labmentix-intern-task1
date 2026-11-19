"""
run_training.py  — Execute model training pipeline and save results.
Run from the Zomato-Sentiment-Analysis directory.
"""
import sys, os, json, warnings
sys.path.insert(0, 'src')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

np.random.seed(42)

from train import (build_preprocessor, split_data, evaluate_model,
                   build_lr_pipeline, build_svc_pipeline,
                   build_rf_pipeline, save_model)
from feature_engineering import (NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
                                  TEXT_FEATURE, TARGET, GROUP_COL)

# ── Load processed data ────────────────────────────────────────────────────────
df = pd.read_csv('data/processed/zomato_merged.csv')
# Ensure text column has no NaN (ColumnTransformer's TfidfVectorizer requires strings)
df['Clean_Review'] = df['Clean_Review'].fillna('').astype(str)
df['Review']       = df['Review'].fillna('').astype(str)
print('Loaded:', df.shape)
print(df['Sentiment'].value_counts())

# ── Split ──────────────────────────────────────────────────────────────────────
df_train, df_test = split_data(df)
X_full_train = df_train[NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TEXT_FEATURE]]
y_train       = df_train[TARGET]
X_full_test   = df_test[NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TEXT_FEATURE]]
y_test        = df_test[TARGET]
label_names   = ['Negative', 'Neutral', 'Positive']

# ── Model 1: Multinomial Naive Bayes (text-only baseline) ─────────────────────
print('\n=== MODEL 1: Naive Bayes (text-only) ===')
nb_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2),
                              min_df=2, max_df=0.95, sublinear_tf=True)),
    ('clf',   MultinomialNB(alpha=0.5))
])
train_text = df_train['Clean_Review'].fillna('').astype(str)
test_text  = df_test['Clean_Review'].fillna('').astype(str)
nb_pipe.fit(train_text, y_train)
m1 = evaluate_model('Naive Bayes', nb_pipe, test_text, y_test, label_names)

# ── Model 2: Logistic Regression (all features) ───────────────────────────────
print('\n=== MODEL 2: Logistic Regression (all features) ===')
prep2    = build_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TEXT_FEATURE)
lr_pipe  = build_lr_pipeline(prep2)
lr_pipe.fit(X_full_train, y_train)
m2 = evaluate_model('Logistic Regression', lr_pipe, X_full_test, y_test, label_names)

# ── Model 3: LinearSVC (all features) ─────────────────────────────────────────
print('\n=== MODEL 3: LinearSVC (all features) ===')
prep3    = build_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TEXT_FEATURE)
svc_pipe = build_svc_pipeline(prep3)
svc_pipe.fit(X_full_train, y_train)
m3 = evaluate_model('LinearSVC', svc_pipe, X_full_test, y_test, label_names)

# ── Model 4: Random Forest (all features) ─────────────────────────────────────
print('\n=== MODEL 4: Random Forest (all features) ===')
prep4    = build_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TEXT_FEATURE)
rf_pipe  = build_rf_pipeline(prep4)
rf_pipe.fit(X_full_train, y_train)
m4 = evaluate_model('Random Forest', rf_pipe, X_full_test, y_test, label_names)

# ── Comparison ─────────────────────────────────────────────────────────────────
results = pd.DataFrame([m1, m2, m3, m4])
print('\n=== MODEL COMPARISON ===')
print(results.to_string(index=False))
os.makedirs('outputs/evaluation', exist_ok=True)
results.to_csv('outputs/evaluation/model_comparison.csv', index=False)

# ── Select best full-feature model ─────────────────────────────────────────────
full_res = results[results['model'] != 'Naive Bayes'].copy()
best_name = full_res.loc[full_res['macro_f1'].idxmax(), 'model']
print(f'\nBest full-feature model: {best_name}')

model_map = {
    'Logistic Regression': lr_pipe,
    'LinearSVC': svc_pipe,
    'Random Forest': rf_pipe,
}
best_pipeline = model_map[best_name]
best_row      = full_res.loc[full_res['model'] == best_name].iloc[0]

meta = {
    'best_model': best_name,
    'macro_f1': float(best_row['macro_f1']),
    'accuracy': float(best_row['accuracy']),
    'balanced_accuracy': float(best_row['balanced_accuracy']),
    'weighted_f1': float(best_row['weighted_f1']),
    'train_rows': int(len(df_train)),
    'test_rows': int(len(df_test)),
    'sentiment_classes': label_names,
    'numerical_features': NUMERICAL_FEATURES,
    'categorical_features': CATEGORICAL_FEATURES,
    'text_feature': TEXT_FEATURE,
}
save_model(best_pipeline, model_name='zomato_sentiment_model', metadata=meta)
print('All done!')
print(json.dumps(meta, indent=2))
