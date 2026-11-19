"""
data_cleaning.py
----------------
Data cleaning and merging utilities for the Zomato Sentiment Analysis project.
"""

import pandas as pd
import numpy as np
import re
import os
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


# ─────────────────────────────────────────────
# LOADING FUNCTIONS
# ─────────────────────────────────────────────

def load_raw_datasets(raw_dir: str = RAW_DIR):
    """Load the two raw CSV files and return them as DataFrames."""
    meta_path = os.path.join(raw_dir, "Zomato Restaurant names and Metadata.csv")
    reviews_path = os.path.join(raw_dir, "Zomato Restaurant reviews.csv")

    for path in [meta_path, reviews_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    meta = pd.read_csv(meta_path)
    reviews = pd.read_csv(reviews_path)

    print(f"[INFO] Metadata loaded   : {meta.shape[0]:>5} rows × {meta.shape[1]} cols")
    print(f"[INFO] Reviews loaded    : {reviews.shape[0]:>5} rows × {reviews.shape[1]} cols")
    return meta, reviews


# ─────────────────────────────────────────────
# NAME NORMALISATION
# ─────────────────────────────────────────────

def normalise_name(name: str) -> str:
    """Lowercase, strip, collapse whitespace, and remove common suffixes."""
    if pd.isna(name):
        return ""
    name = str(name).strip().lower()
    name = re.sub(r"\s+", " ", name)
    return name


# ─────────────────────────────────────────────
# MERGE
# ─────────────────────────────────────────────

def merge_datasets(meta: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """
    LEFT-JOIN reviews → metadata on restaurant name.

    Why LEFT JOIN?
    Every review is a data-point for supervised learning.  A LEFT JOIN keeps
    all 10 000 reviews as rows even when a restaurant is not in the metadata
    table (e.g., small data-entry gap), so no labelled observations are
    silently lost.  The five metadata restaurants that have no reviews are
    simply absent from the result — they cannot contribute review-level labels.
    """
    # Preserve originals and create normalised keys
    meta = meta.copy()
    reviews = reviews.copy()

    meta["_name_key"] = meta["Name"].apply(normalise_name)
    reviews["_rest_key"] = reviews["Restaurant"].apply(normalise_name)

    merged = reviews.merge(
        meta,
        left_on="_rest_key",
        right_on="_name_key",
        how="left",
        validate="many_to_one",   # each review maps to at most one metadata row
        indicator=True,
    )

    # Merge diagnostics
    merge_counts = merged["_merge"].value_counts()
    print(f"\n[MERGE DIAGNOSTICS]")
    print(f"  Both matched   : {merge_counts.get('both', 0)}")
    print(f"  Left only      : {merge_counts.get('left_only', 0)}  ← reviews without metadata")

    # Drop redundant columns
    merged.drop(columns=["Name", "_name_key", "_rest_key", "_merge", "Links"],
                inplace=True, errors="ignore")

    print(f"\n[INFO] Merged dataset   : {merged.shape[0]} rows × {merged.shape[1]} cols")
    return merged


# ─────────────────────────────────────────────
# COST CLEANING
# ─────────────────────────────────────────────

def clean_cost(series: pd.Series) -> pd.Series:
    """Remove commas, coerce to float, return numeric series."""
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .pipe(pd.to_numeric, errors="coerce")
    )


# ─────────────────────────────────────────────
# RATING CLEANING
# ─────────────────────────────────────────────

def clean_rating(series: pd.Series) -> pd.Series:
    """
    Convert Rating to numeric.
    - 'Like' → NaN  (not a numeric rating, cannot be mapped to 1-5 scale)
    - Ratings outside [1, 5] → NaN
    """
    numeric = pd.to_numeric(series, errors="coerce")
    # Mark out-of-range values as NaN
    numeric = numeric.where((numeric >= 1) & (numeric <= 5))
    return numeric


# ─────────────────────────────────────────────
# SENTIMENT TARGET
# ─────────────────────────────────────────────

def classify_sentiment(rating: float) -> str:
    """Map numeric rating to three-class sentiment label."""
    if pd.isna(rating):
        return np.nan
    if rating <= 2:
        return "Negative"
    elif rating < 4:
        return "Neutral"
    else:
        return "Positive"


# ─────────────────────────────────────────────
# DATETIME FEATURES
# ─────────────────────────────────────────────

def extract_time_features(df: pd.DataFrame, time_col: str = "Time") -> pd.DataFrame:
    """Parse the Time column and extract date/time components."""
    df = df.copy()
    df["Review_Datetime"] = pd.to_datetime(df[time_col], errors="coerce")
    df["Review_Year"]  = df["Review_Datetime"].dt.year
    df["Review_Month"] = df["Review_Datetime"].dt.month
    df["Review_Day"]   = df["Review_Datetime"].dt.day
    df["Review_DOW"]   = df["Review_Datetime"].dt.dayofweek   # 0=Mon, 6=Sun
    df["Review_Hour"]  = df["Review_Datetime"].dt.hour
    return df


# ─────────────────────────────────────────────
# REVIEWER METADATA
# ─────────────────────────────────────────────

def extract_reviewer_meta(series: pd.Series) -> pd.DataFrame:
    """
    Parse 'X Reviews , Y Followers' string into numeric columns.
    Returns a 2-column DataFrame: Reviewer_Reviews, Reviewer_Followers.
    """
    parsed = series.str.extract(
        r"(?P<Reviewer_Reviews>\d+)\s+Review[s]?\s*,\s*(?P<Reviewer_Followers>\d+)\s+Follower",
        expand=True,
    )
    return parsed.apply(pd.to_numeric, errors="coerce")


# ─────────────────────────────────────────────
# FULL CLEANING PIPELINE
# ─────────────────────────────────────────────

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps to the merged DataFrame.
    Returns a cleaned DataFrame ready for EDA and feature engineering.
    """
    df = df.copy()

    # ── 1. Cost
    df["Cost"] = clean_cost(df["Cost"])
    cost_median = df["Cost"].median()
    df["Cost"].fillna(cost_median, inplace=True)
    print(f"[CLEAN] Cost  — {df['Cost'].isna().sum()} NaN remaining  (median={cost_median})")

    # ── 2. Rating → numeric, then drop rows with no valid target
    df["Rating"] = clean_rating(df["Rating"])
    before = len(df)
    df.dropna(subset=["Rating"], inplace=True)
    after = len(df)
    print(f"[CLEAN] Rating — dropped {before - after} rows with invalid/missing rating")

    # ── 3. Sentiment target
    df["Sentiment"] = df["Rating"].apply(classify_sentiment)

    # ── 4. Duplicates — exact row duplicates
    n_dup = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    print(f"[CLEAN] Duplicates — removed {n_dup} exact duplicates")

    # ── 5. Time features
    df = extract_time_features(df, "Time")

    # ── 6. Reviewer metadata
    rev_meta = extract_reviewer_meta(df["Metadata"])
    df["Reviewer_Reviews"]   = rev_meta["Reviewer_Reviews"].fillna(1)
    df["Reviewer_Followers"] = rev_meta["Reviewer_Followers"].fillna(0)

    # ── 7. Review text — missing handling
    df["Review"] = df["Review"].fillna("")
    df["Review"] = df["Review"].astype(str).str.strip()

    # ── 8. Missing categoricals
    df["Collections"] = df["Collections"].fillna("Not Available")
    df["Timings"]     = df["Timings"].fillna("Not Available")
    df["Cuisines"]    = df["Cuisines"].fillna("Not Available")
    df["Reviewer"]    = df["Reviewer"].fillna("Anonymous")

    # ── 9. Normalise strings
    df["Restaurant"] = df["Restaurant"].str.strip()
    df["Cuisines"]   = df["Cuisines"].str.strip()

    # ── 10. Reset index
    df.reset_index(drop=True, inplace=True)

    print(f"\n[CLEAN] Final dataset   : {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"[CLEAN] Sentiment dist  :\n{df['Sentiment'].value_counts()}")
    return df


# ─────────────────────────────────────────────
# SAVE HELPER
# ─────────────────────────────────────────────

def save_processed(df: pd.DataFrame, processed_dir: str = PROCESSED_DIR) -> str:
    """Save cleaned merged DataFrame to data/processed/zomato_merged.csv."""
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, "zomato_merged.csv")
    df.to_csv(out_path, index=False)
    print(f"[SAVE] Processed dataset saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────
# ENTRY POINT (for standalone testing)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    meta, reviews = load_raw_datasets()
    merged = merge_datasets(meta, reviews)
    cleaned = clean_dataset(merged)
    save_processed(cleaned)
