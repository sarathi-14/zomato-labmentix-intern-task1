"""
feature_engineering.py
-----------------------
Builds numerical, categorical, and text features from the cleaned dataset.
"""

import pandas as pd
import numpy as np
import re
import os


# ─────────────────────────────────────────────
# REVIEW-LEVEL TEXT FEATURES
# ─────────────────────────────────────────────

def add_text_features(df: pd.DataFrame,
                      review_col: str = "Review",
                      clean_col: str = "Clean_Review") -> pd.DataFrame:
    """Add length-based text features derived from review content."""
    df = df.copy()
    df["Review_Length"]    = df[review_col].str.len().fillna(0).astype(int)
    df["Word_Count"]       = df[review_col].apply(
        lambda x: len(str(x).split()) if pd.notna(x) and str(x).strip() else 0
    )
    df["Char_Count"]       = df[review_col].apply(
        lambda x: len(re.sub(r"\s", "", str(x))) if pd.notna(x) else 0
    )
    df["Clean_Word_Count"] = df[clean_col].apply(
        lambda x: len(str(x).split()) if pd.notna(x) and str(x).strip() else 0
    )
    return df


# ─────────────────────────────────────────────
# PRIMARY CUISINE EXTRACTION
# ─────────────────────────────────────────────

def extract_primary_cuisine(cuisines: str) -> str:
    """Extract the first listed cuisine as the primary cuisine."""
    if pd.isna(cuisines) or str(cuisines).strip() == "":
        return "Not Available"
    return str(cuisines).split(",")[0].strip()


def add_cuisine_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Primary_Cuisine  : first cuisine listed
    - Cuisine_Count    : number of cuisines offered
    """
    df = df.copy()
    df["Primary_Cuisine"] = df["Cuisines"].apply(extract_primary_cuisine)
    df["Cuisine_Count"]   = df["Cuisines"].apply(
        lambda x: len(str(x).split(",")) if pd.notna(x) and x != "Not Available" else 1
    )
    return df


# ─────────────────────────────────────────────
# COST CATEGORY
# ─────────────────────────────────────────────

def add_cost_category(df: pd.DataFrame, cost_col: str = "Cost") -> pd.DataFrame:
    """
    Bin restaurant cost into Budget / Mid-range / Premium categories.
    Thresholds derived from quantiles of the actual cost distribution.
    """
    df = df.copy()
    # Use actual data quantiles for binning
    q33 = df[cost_col].quantile(0.33)
    q66 = df[cost_col].quantile(0.66)
    bins   = [0, q33, q66, float("inf")]
    labels = ["Budget", "Mid-Range", "Premium"]
    df["Cost_Category"] = pd.cut(
        df[cost_col], bins=bins, labels=labels, include_lowest=True
    ).astype(str)
    df["Cost_Category"].fillna("Unknown", inplace=True)
    print(f"[FE] Cost Category thresholds: Budget ≤ {q33:.0f}, Mid-Range ≤ {q66:.0f}, Premium > {q66:.0f}")
    return df


# ─────────────────────────────────────────────
# COLLECTIONS FLAG
# ─────────────────────────────────────────────

def add_collection_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Has_Collection : binary flag (1 if the restaurant is in any curated collection)
    - Collection_Count : approximate number of collections
    """
    df = df.copy()
    df["Has_Collection"] = df["Collections"].apply(
        lambda x: 0 if (pd.isna(x) or x == "Not Available") else 1
    )
    df["Collection_Count"] = df["Collections"].apply(
        lambda x: 0 if (pd.isna(x) or x == "Not Available")
        else len(str(x).split(","))
    )
    return df


# ─────────────────────────────────────────────
# TIMING CATEGORY
# ─────────────────────────────────────────────

def classify_timing(timing: str) -> str:
    """
    Heuristically classify restaurant timing into:
    All Day / Lunch & Dinner / Lunch Only / Dinner Only / Unknown.
    """
    if pd.isna(timing) or timing == "Not Available":
        return "Unknown"
    t = timing.lower()
    if "24" in t or "all day" in t:
        return "All Day"
    has_am = bool(re.search(r"\b(am|noon|12)\b", t))
    has_pm = bool(re.search(r"\b(pm|evening|night)\b", t))
    if has_am and has_pm:
        return "Lunch and Dinner"
    elif has_am:
        return "Lunch Only"
    elif has_pm:
        return "Dinner Only"
    return "Other"


def add_timing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Timing_Category column."""
    df = df.copy()
    df["Timing_Category"] = df["Timings"].apply(classify_timing)
    return df


# ─────────────────────────────────────────────
# FULL FEATURE ENGINEERING PIPELINE
# ─────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations.
    Call this AFTER data_cleaning.clean_dataset() and
    preprocessing.apply_text_preprocessing().
    """
    df = add_text_features(df)
    df = add_cuisine_features(df)
    df = add_cost_category(df)
    df = add_collection_features(df)
    df = add_timing_features(df)

    # Encode sentiment as numeric for some uses (not used as model input)
    sentiment_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
    df["Sentiment_Code"] = df["Sentiment"].map(sentiment_map)

    print(f"[FE] Feature-engineered dataset: {df.shape[0]} rows × {df.shape[1]} cols")
    return df


# ─────────────────────────────────────────────
# FEATURE DEFINITIONS (used by train.py)
# ─────────────────────────────────────────────

NUMERICAL_FEATURES = [
    "Cost",
    "Pictures",
    "Review_Length",
    "Word_Count",
    "Char_Count",
    "Review_Year",
    "Review_Month",
    "Review_Hour",
    "Reviewer_Reviews",
    "Reviewer_Followers",
    "Cuisine_Count",
    "Collection_Count",
]

CATEGORICAL_FEATURES = [
    "Primary_Cuisine",
    "Cost_Category",
    "Timing_Category",
]

TEXT_FEATURE = "Clean_Review"

TARGET = "Sentiment"
GROUP_COL = "Restaurant"

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_cleaning import load_raw_datasets, merge_datasets, clean_dataset, save_processed
    from preprocessing import apply_text_preprocessing

    meta, reviews = load_raw_datasets()
    merged  = merge_datasets(meta, reviews)
    cleaned = clean_dataset(merged)
    cleaned = apply_text_preprocessing(cleaned)
    featured = build_features(cleaned)
    print(featured.dtypes)
    print(featured[NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET]].head(3))
