"""
train.py
--------
Model training, hyperparameter tuning, evaluation, and model saving.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Scikit-learn imports ──────────────────────────────────────────────────────
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix,
    f1_score, make_scorer
)

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from feature_engineering import (
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TEXT_FEATURE, TARGET, GROUP_COL
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# PREPROCESSING PIPELINE BUILDER
# ─────────────────────────────────────────────

def build_preprocessor(numerical_features, categorical_features, text_feature,
                        tfidf_max_features=5000, ngram_range=(1, 2)):
    """Build a ColumnTransformer for numerical, categorical, and text features."""

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    text_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=tfidf_max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num",  num_pipeline,  numerical_features),
            ("cat",  cat_pipeline,  categorical_features),
            ("text", text_pipeline, text_feature),
        ],
        remainder="drop",
        n_jobs=1,
    )
    return preprocessor


# ─────────────────────────────────────────────
# DATA SPLITTING (restaurant-grouped)
# ─────────────────────────────────────────────

def split_data(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42):
    """
    Restaurant-grouped train/test split.
    GroupShuffleSplit ensures NO restaurant appears in both train and test.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    groups = df[GROUP_COL]

    train_idx, test_idx = next(gss.split(df, df[TARGET], groups=groups))

    df_train = df.iloc[train_idx].copy()
    df_test  = df.iloc[test_idx].copy()

    # Verify no overlap
    train_restaurants = set(df_train[GROUP_COL].unique())
    test_restaurants  = set(df_test[GROUP_COL].unique())
    overlap = train_restaurants & test_restaurants
    assert len(overlap) == 0, f"Restaurant overlap between train/test! {overlap}"

    print(f"[SPLIT] Train : {len(df_train)} rows from {len(train_restaurants)} restaurants")
    print(f"[SPLIT] Test  : {len(df_test)}  rows from {len(test_restaurants)}  restaurants")
    print(f"[SPLIT] No restaurant overlap ✓")
    return df_train, df_test


# ─────────────────────────────────────────────
# EVALUATION HELPER
# ─────────────────────────────────────────────

def evaluate_model(name: str, pipeline, X_test, y_test, label_names=None) -> dict:
    """Evaluate a fitted pipeline and return a metrics dictionary."""
    y_pred = pipeline.predict(X_test)

    metrics = {
        "model": name,
        "accuracy":          round(accuracy_score(y_test, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_test, y_pred), 4),
        "macro_f1":          round(f1_score(y_test, y_pred, average="macro"), 4),
        "weighted_f1":       round(f1_score(y_test, y_pred, average="weighted"), 4),
    }

    print(f"\n{'='*55}")
    print(f" Model : {name}")
    print(f"{'='*55}")
    print(f"  Accuracy          : {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy : {metrics['balanced_accuracy']:.4f}")
    print(f"  Macro F1          : {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1       : {metrics['weighted_f1']:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=label_names)}")
    return metrics


# ─────────────────────────────────────────────
# MODEL BUILDERS
# ─────────────────────────────────────────────

def build_nb_pipeline(preprocessor):
    """Multinomial Naive Bayes — text-only pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2),
                                  min_df=2, max_df=0.95, sublinear_tf=True)),
        ("clf",   MultinomialNB(alpha=0.5)),
    ])


def build_lr_pipeline(preprocessor):
    return Pipeline([
        ("prep", preprocessor),
        ("clf",  LogisticRegression(C=1.0, max_iter=1000,
                                    class_weight="balanced",
                                    random_state=42, solver="lbfgs")),
    ])


def build_svc_pipeline(preprocessor):
    return Pipeline([
        ("prep", preprocessor),
        ("clf",  LinearSVC(C=1.0, class_weight="balanced",
                           max_iter=2000, random_state=42)),
    ])


def build_rf_pipeline(preprocessor):
    return Pipeline([
        ("prep", preprocessor),
        ("clf",  RandomForestClassifier(n_estimators=200, max_depth=15,
                                         class_weight="balanced",
                                         random_state=42, n_jobs=-1)),
    ])


# ─────────────────────────────────────────────
# HYPERPARAMETER TUNING
# ─────────────────────────────────────────────

def tune_model(pipeline, param_grid: dict, df_train: pd.DataFrame,
               X_col: str = None, text_only: bool = False,
               n_iter: int = 20, random_state: int = 42):
    """
    RandomizedSearchCV with GroupKFold (5 folds).
    Uses Macro F1 as the scoring metric (handles class imbalance better than accuracy).
    """
    groups = df_train[GROUP_COL]
    if text_only:
        X_train = df_train["Clean_Review"]
    else:
        X_train = df_train[NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TEXT_FEATURE]]

    y_train = df_train[TARGET]

    gkf = GroupKFold(n_splits=5)
    scorer = make_scorer(f1_score, average="macro")

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=gkf.split(X_train, y_train, groups=groups),
        scoring=scorer,
        refit=True,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    print(f"  Best Macro F1 (CV) : {search.best_score_:.4f}")
    print(f"  Best params        : {search.best_params_}")
    return search.best_estimator_


# ─────────────────────────────────────────────
# SAVE / LOAD MODEL
# ─────────────────────────────────────────────

def save_model(pipeline, model_name: str = "zomato_sentiment_model",
               metadata: dict = None, models_dir: str = MODELS_DIR):
    """Save fitted pipeline and optional metadata JSON."""
    model_path = os.path.join(models_dir, f"{model_name}.joblib")
    joblib.dump(pipeline, model_path)
    print(f"[SAVE] Model saved → {model_path}")

    if metadata:
        meta_path = os.path.join(models_dir, "model_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[SAVE] Metadata saved → {meta_path}")

    return model_path


def load_model(model_name: str = "zomato_sentiment_model",
               models_dir: str = MODELS_DIR):
    """Load a saved joblib pipeline."""
    model_path = os.path.join(models_dir, f"{model_name}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    pipeline = joblib.load(model_path)
    print(f"[LOAD] Model loaded ← {model_path}")
    return pipeline


if __name__ == "__main__":
    print("train.py loaded successfully. Import and call individual functions from the notebook.")
