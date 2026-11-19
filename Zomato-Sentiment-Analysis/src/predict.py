"""
predict.py
----------
Prediction script for Zomato Sentiment Analysis.
Loads the saved model and predicts on new unseen review data.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from feature_engineering import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TEXT_FEATURE
from preprocessing import preprocess_review

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_pipeline(model_name: str = "zomato_sentiment_model"):
    """Load the saved model pipeline."""
    path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")
    return joblib.load(path)


def build_input_df(review: str, cost: float = 800.0, cuisine: str = "North Indian",
                   collections: str = "Not Available", pictures: int = 0,
                   review_year: int = 2019, review_month: int = 5,
                   review_hour: int = 14) -> pd.DataFrame:
    """
    Build a single-row input DataFrame with all required features.
    This mirrors the feature engineering applied during training.
    """
    clean = preprocess_review(review)

    row = {
        # Numerical features
        "Cost":               cost,
        "Pictures":           pictures,
        "Review_Length":      len(review),
        "Word_Count":         len(review.split()),
        "Char_Count":         len(review.replace(" ", "")),
        "Review_Year":        review_year,
        "Review_Month":       review_month,
        "Review_Hour":        review_hour,
        "Reviewer_Reviews":   1,
        "Reviewer_Followers": 0,
        "Cuisine_Count":      len(cuisine.split(",")),
        "Collection_Count":   0 if collections == "Not Available" else len(collections.split(",")),

        # Categorical features
        "Primary_Cuisine":  cuisine.split(",")[0].strip(),
        "Cost_Category":    _get_cost_category(cost),
        "Timing_Category":  "Lunch and Dinner",

        # Text feature
        "Clean_Review": clean,
    }

    return pd.DataFrame([row])


def _get_cost_category(cost: float) -> str:
    """Simple heuristic cost categorisation matching training thresholds."""
    if cost <= 500:
        return "Budget"
    elif cost <= 1000:
        return "Mid-Range"
    else:
        return "Premium"


def predict_sentiment(review: str, cost: float = 800.0, cuisine: str = "North Indian",
                      collections: str = "Not Available", pictures: int = 0,
                      model_name: str = "zomato_sentiment_model") -> dict:
    """
    Predict sentiment for a new review and return a result dictionary.

    Returns:
        dict with keys: sentiment, confidence (if available), input_summary
    """
    pipeline = load_pipeline(model_name)
    input_df = build_input_df(review, cost, cuisine, collections, pictures)

    prediction = pipeline.predict(input_df)[0]

    result = {"sentiment": prediction, "input_review": review}

    # Confidence score (not all models support predict_proba)
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(input_df)[0]
            classes = pipeline.classes_
            result["confidence"] = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
        except Exception:
            result["confidence"] = None
    else:
        result["confidence"] = None

    return result


# ─────────────────────────────────────────────
# SAMPLE PREDICTIONS (run directly)
# ─────────────────────────────────────────────

UNSEEN_REVIEWS = [
    {
        "review":      "Absolutely loved the food! The biryani was amazing and service was top-notch.",
        "cost":        1200,
        "cuisine":     "Biryani, North Indian",
        "collections": "Top-Rated, Gold Curated",
        "pictures":    3,
        "expected":    "Positive",
    },
    {
        "review":      "Terrible experience. Food was cold, service was rude and we had to wait for 45 minutes.",
        "cost":        800,
        "cuisine":     "Chinese, Continental",
        "collections": "Not Available",
        "pictures":    0,
        "expected":    "Negative",
    },
    {
        "review":      "Average food. Nothing special but not bad either. Decent place for a quick meal.",
        "cost":        500,
        "cuisine":     "South Indian",
        "collections": "Not Available",
        "pictures":    1,
        "expected":    "Neutral",
    },
    {
        "review":      "Not worth the price. The ambience was okay but food quality was not satisfactory.",
        "cost":        1500,
        "cuisine":     "Continental",
        "collections": "Great Buffets",
        "pictures":    0,
        "expected":    "Negative",
    },
    {
        "review":      "Great place to hang out with family. Food was delicious and staff was very helpful.",
        "cost":        900,
        "cuisine":     "North Indian, Chinese",
        "collections": "Corporate Favorites",
        "pictures":    5,
        "expected":    "Positive",
    },
]


if __name__ == "__main__":
    print("=" * 60)
    print("  ZOMATO SENTIMENT PREDICTOR — Unseen Review Predictions")
    print("=" * 60)

    all_correct = 0
    for i, sample in enumerate(UNSEEN_REVIEWS, 1):
        result = predict_sentiment(
            review=sample["review"],
            cost=sample["cost"],
            cuisine=sample["cuisine"],
            collections=sample["collections"],
            pictures=sample["pictures"],
        )
        correct = "✓" if result["sentiment"] == sample["expected"] else "✗"
        print(f"\n[{i}] Review   : {sample['review'][:70]}…")
        print(f"    Expected : {sample['expected']}")
        print(f"    Predicted: {result['sentiment']}  {correct}")
        if result["confidence"]:
            print(f"    Confidence: {result['confidence']}")
        all_correct += (result["sentiment"] == sample["expected"])

    print(f"\n[RESULT] Correct predictions: {all_correct}/{len(UNSEEN_REVIEWS)}")
