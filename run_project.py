"""
Zomato Review Rating Prediction Project
End-to-End Pipeline: Merge, Clean, Train, Evaluate, and Save Model
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def run_pipeline():
    print("=" * 70)
    print("   ZOMATO RESTAURANT REVIEWS: RATING PREDICTION PIPELINE")
    print("=" * 70)

    # 1. Locate and Load Datasets
    print("\n[Step 1/7] Locating and loading datasets...")
    files = os.listdir('.')
    review_file = next((f for f in files if 'review' in f.lower() and f.lower().endswith('.csv')), 'Zomato Restaurant reviews.csv')
    metadata_file = next((f for f in files if ('metadata' in f.lower() or 'names' in f.lower()) and f.lower().endswith('.csv')), 'Zomato Restaurant names and Metadata.csv')

    if not os.path.exists(review_file) or not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Missing required CSV files: '{review_file}' or '{metadata_file}'")

    print(f" - Reviews dataset: {review_file}")
    print(f" - Metadata dataset: {metadata_file}")

    reviews = pd.read_csv(review_file)
    restaurants = pd.read_csv(metadata_file)
    print(f" - Raw reviews shape: {reviews.shape[0]:,} rows, {reviews.shape[1]} columns")
    print(f" - Raw restaurant metadata shape: {restaurants.shape[0]:,} rows, {restaurants.shape[1]} columns")

    # 2. Standardize Columns and Merge
    print("\n[Step 2/7] Standardizing columns and merging...")
    reviews.columns = reviews.columns.str.strip()
    restaurants.columns = restaurants.columns.str.strip()

    required_reviews = {'Restaurant', 'Review', 'Rating'}
    required_metadata = {'Name'}
    if not required_reviews.issubset(reviews.columns):
        raise ValueError(f"Reviews missing required columns: {required_reviews - set(reviews.columns)}")
    if not required_metadata.issubset(restaurants.columns):
        raise ValueError(f"Metadata missing required columns: {required_metadata - set(restaurants.columns)}")

    restaurants_dedup = restaurants.drop_duplicates(subset='Name', keep='first').copy()
    merged = reviews.merge(restaurants_dedup, left_on='Restaurant', right_on='Name', how='left', indicator=True)
    match_counts = merged['_merge'].value_counts()
    print(f" - Successfully merged {match_counts.get('both', 0):,} records.")
    merged = merged.drop(columns=['Name', '_merge'], errors='ignore')

    # 3. Clean and Prepare Fields
    print("\n[Step 3/7] Cleaning and preparing features...")
    merged['Rating'] = pd.to_numeric(merged['Rating'], errors='coerce')
    merged['Review'] = merged['Review'].fillna('').astype(str).str.strip()

    model_data = merged.dropna(subset=['Rating']).copy()
    model_data = model_data[model_data['Review'].ne('')].copy()
    rows_before_dedup = len(model_data)
    model_data = model_data.drop_duplicates().reset_index(drop=True)
    dropped_rows = len(merged) - len(model_data)
    print(f" - Retained {len(model_data):,} rows (filtered out {dropped_rows:,} rows with missing rating/empty review/duplicates)")

    # Parse Cost, Pictures, and Datetime
    if 'Cost' in model_data.columns:
        model_data['Cost'] = pd.to_numeric(model_data['Cost'].astype(str).str.replace(',', '', regex=False), errors='coerce')
    if 'Pictures' in model_data.columns:
        model_data['Pictures'] = pd.to_numeric(model_data['Pictures'], errors='coerce')

    if 'Time' in model_data.columns:
        parsed_time = pd.to_datetime(model_data['Time'], errors='coerce')
        model_data['ReviewHour'] = parsed_time.dt.hour
        model_data['ReviewMonth'] = parsed_time.dt.month
        model_data['ReviewDayOfWeek'] = parsed_time.dt.dayofweek

    # 4. Predictors Selection and Grouped Splitting
    print("\n[Step 4/7] Selecting predictors and splitting dataset...")
    text_column = 'Review'
    categorical_candidates = ['Restaurant', 'Cuisines', 'Collections', 'Timings']
    numeric_candidates = ['Cost', 'Pictures', 'ReviewHour', 'ReviewMonth', 'ReviewDayOfWeek']
    categorical_columns = [c for c in categorical_candidates if c in model_data.columns]
    numeric_columns = [c for c in numeric_candidates if c in model_data.columns]

    X = model_data[[text_column] + categorical_columns + numeric_columns].copy()
    y = model_data['Rating'].astype(float)
    groups = model_data['Restaurant'].astype(str)

    if groups.nunique() >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        split_type = "Restaurant-grouped 80/20 split (Holdout restaurants unseen in training)"
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        split_type = "Random 80/20 train/test split"

    print(f" - Split strategy: {split_type}")
    print(f" - Training samples: {len(X_train):,} ({groups.iloc[train_idx].nunique()} unique restaurants)")
    print(f" - Testing samples:  {len(X_test):,} ({groups.iloc[test_idx].nunique()} unique restaurants)")

    # 5. Build and Train Machine Learning Pipeline
    print("\n[Step 5/7] Building and training ML regression pipeline...")
    text_transformer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words='english',
        sublinear_tf=True
    )
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scale', StandardScaler(with_mean=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('review_text', text_transformer, text_column),
        ('categories', categorical_transformer, categorical_columns),
        ('numbers', numeric_transformer, numeric_columns)
    ])

    model = Pipeline(steps=[
        ('features', preprocessor),
        ('regressor', Ridge(alpha=2.0, solver='lsqr'))
    ])

    model.fit(X_train, y_train)
    print(" - Pipeline model trained successfully.")

    # 6. Evaluation
    print("\n[Step 6/7] Evaluating model performance on test set...")
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    baseline_mae = mean_absolute_error(y_test, np.repeat(y_train.median(), len(y_test)))

    print("-" * 50)
    print(f"   Model Evaluation Metrics:")
    print(f"   - Mean Absolute Error (MAE)  : {mae:.4f} rating points")
    print(f"   - Root Mean Sq Error (RMSE)  : {rmse:.4f} rating points")
    print(f"   - R-squared (R²)             : {r2:.4f}")
    print(f"   - Baseline MAE (Median Pred) : {baseline_mae:.4f} rating points")
    print(f"   - MAE Improvement vs Baseline: {(1 - mae / baseline_mae) * 100:.2f}%")
    print("-" * 50)

    # Save visualization plot
    plot_filename = 'actual_vs_predicted.png'
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, predictions, alpha=0.3, color='#1f77b4', edgecolors='none', label='Test Predictions')
    plt.plot([1, 5], [1, 5], color='red', linestyle='--', linewidth=2, label='Perfect Prediction Line')
    plt.xlim(0.8, 5.2)
    plt.ylim(0.8, 5.2)
    plt.xlabel('Actual Review Rating', fontsize=11)
    plt.ylabel('Predicted Review Rating', fontsize=11)
    plt.title('Actual vs Predicted Zomato Ratings (Grouped Holdout)', fontsize=13)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True)
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" - Evaluation chart saved as: {plot_filename}")

    # Inspect sample predictions
    results = X_test[[text_column] + categorical_columns].copy()
    results['ActualRating'] = y_test.values
    results['PredictedRating'] = np.round(predictions, 2)
    results['AbsoluteError'] = np.round(np.abs(results['ActualRating'] - results['PredictedRating']), 2)
    print("\nSample Predictions Preview (Top 5 lowest error):")
    sample_preview = results.sort_values('AbsoluteError').head(5)
    for idx, row in sample_preview.iterrows():
        review_snippet = (row['Review'][:60] + '...') if len(row['Review']) > 60 else row['Review']
        print(f"   Restaurant: {row['Restaurant']:<25} | Actual: {row['ActualRating']} | Predicted: {row['PredictedRating']:<4} | Error: {row['AbsoluteError']}")
        print(f"     Review: \"{review_snippet}\"")

    # 7. Save Artifacts
    print("\n[Step 7/7] Exporting cleaned dataset and model pipeline...")
    cleaned_output = 'zomato_merged_cleaned.csv'
    model_output = 'zomato_rating_model.joblib'
    model_data.to_csv(cleaned_output, index=False)
    joblib.dump(model, model_output)
    print(f" - Cleaned data saved:  {cleaned_output} ({os.path.getsize(cleaned_output):,} bytes)")
    print(f" - Trained model saved: {model_output} ({os.path.getsize(model_output):,} bytes)")
    print("\n" + "=" * 70)
    print("   PROJECT EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == '__main__':
    run_pipeline()
