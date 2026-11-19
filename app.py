"""
app.py
------
Streamlit web application for Zomato Customer Sentiment Prediction.

Run with:
    streamlit run app.py
"""

import os
import sys
import warnings
import streamlit as st
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zomato Sentiment Predictor",
    page_icon="🍽️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Ensure src is on the path ─────────────────────────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
for cand in [
    os.path.join(APP_DIR, "src"),
    os.path.join(APP_DIR, "Zomato-Sentiment-Analysis", "src"),
]:
    if os.path.exists(cand) and cand not in sys.path:
        sys.path.insert(0, cand)

try:
    from preprocessing import preprocess_review
    from predict import load_pipeline, build_input_df, _get_cost_category
except ImportError as e:
    st.error(f"Could not import project modules: {e}\nMake sure you are running from the project root.")
    st.stop()

# ── Load model (cached) ───────────────────────────────────────────────────────
MODEL_PATH = os.path.join(APP_DIR, "models", "zomato_sentiment_model.joblib")
if not os.path.exists(MODEL_PATH):
    alt_path = os.path.join(APP_DIR, "Zomato-Sentiment-Analysis", "models", "zomato_sentiment_model.joblib")
    if os.path.exists(alt_path):
        MODEL_PATH = alt_path


@st.cache_resource
def get_model():
    """Load the saved sentiment model pipeline (cached for efficiency)."""
    if not os.path.exists(MODEL_PATH):
        return None
    return load_pipeline()


model = get_model()

# ── Sentiment colours & emojis ────────────────────────────────────────────────
SENTIMENT_STYLE = {
    "Positive": {"emoji": "😊", "color": "#28a745", "badge": "SUCCESS"},
    "Neutral":  {"emoji": "😐", "color": "#ffc107", "badge": "INFO"},
    "Negative": {"emoji": "😞", "color": "#dc3545", "badge": "ERROR"},
}

CUISINE_OPTIONS = [
    "North Indian", "South Indian", "Chinese", "Continental", "Biryani",
    "Italian", "Mexican", "Japanese", "Thai", "Mediterranean",
    "Mughlai", "Kebab", "Desserts", "Bakery", "Fast Food",
    "Asian", "Seafood", "Street Food", "Beverages", "Other",
]

COLLECTION_OPTIONS = [
    "Not Available",
    "Top-Rated",
    "Gold Curated",
    "Great Buffets",
    "Corporate Favorites",
    "Hyderabad's Hottest",
    "Food Hygiene Rated Restaurants in Hyderabad",
    "Live Sports Screenings",
    "Date Night",
    "Family Friendly",
]

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("🍽️ Zomato Customer Sentiment Predictor")
st.markdown(
    """
    > **Predict the sentiment** of a customer review for a Zomato restaurant  
    > using a Machine Learning model trained on 10,000 real Zomato reviews.
    """
)

if model is None:
    st.error(
        "⚠️ **Model file not found.**  \n"
        "Please train the model first by running the Jupyter Notebook or:  \n"
        "```bash\n"
        "cd src && python train.py\n"
        "```"
    )
    st.stop()

st.success("✅ Model loaded successfully!")

# ─────────────────────────────────────────────
# SIDEBAR — Restaurant Context
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("🏪 Restaurant Details")
    st.markdown("Provide context about the restaurant to improve prediction accuracy.")

    cost = st.number_input(
        "💰 Average Cost for Two (₹)",
        min_value=100,
        max_value=5000,
        value=800,
        step=100,
        help="Average meal cost for two people in Indian Rupees.",
    )

    cuisine = st.selectbox(
        "🍜 Primary Cuisine",
        options=CUISINE_OPTIONS,
        index=0,
        help="Select the restaurant's primary cuisine type.",
    )

    collection = st.selectbox(
        "🏆 Restaurant Collection",
        options=COLLECTION_OPTIONS,
        index=0,
        help="Curated collection the restaurant belongs to, if any.",
    )

    pictures = st.slider(
        "📸 Number of Photos Attached",
        min_value=0,
        max_value=50,
        value=0,
        help="Number of photos the reviewer attached with their review.",
    )

    st.markdown("---")
    st.caption("ℹ️ **Note:** All predictions are model-generated estimates based on historical review patterns.")

# ─────────────────────────────────────────────
# MAIN — Review Input
# ─────────────────────────────────────────────
st.header("✍️ Customer Review")

review_text = st.text_area(
    label="Enter your restaurant review:",
    placeholder="e.g., 'The food was amazing! Great ambience and very helpful staff. Will definitely visit again.'",
    height=160,
    max_chars=2000,
    help="Enter the full customer review text. The model works best with reviews of 10+ words.",
)

col1, col2 = st.columns([1, 3])
with col1:
    predict_btn = st.button("🔍 Predict Sentiment", type="primary", use_container_width=True)
with col2:
    if st.button("🗑️ Clear", use_container_width=True):
        st.rerun()

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
if predict_btn:
    # ── Validation ────────────────────────────
    if not review_text or len(review_text.strip()) < 5:
        st.warning("⚠️ Please enter a review of at least 5 characters.")
        st.stop()

    if len(review_text.split()) < 3:
        st.warning("⚠️ For best results, please enter a review with at least 3 words.")

    with st.spinner("Analysing sentiment…"):
        try:
            # Build feature DataFrame
            input_df = build_input_df(
                review=review_text,
                cost=float(cost),
                cuisine=cuisine,
                collections=collection,
                pictures=int(pictures),
            )

            # Predict
            prediction = model.predict(input_df)[0]
            style = SENTIMENT_STYLE.get(prediction, SENTIMENT_STYLE["Neutral"])

            # Confidence (if available)
            confidence_data = None
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(input_df)[0]
                    classes = model.classes_
                    confidence_data = dict(zip(classes, proba))
                except Exception:
                    confidence_data = None

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.stop()

    # ── Results ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("📊 Prediction Result")

    result_col1, result_col2 = st.columns([1, 2])

    with result_col1:
        emoji = style["emoji"]
        color = style["color"]
        st.markdown(
            f"""
            <div style="
                background-color: {color}22;
                border: 2px solid {color};
                border-radius: 12px;
                padding: 24px;
                text-align: center;
            ">
                <div style="font-size: 60px;">{emoji}</div>
                <div style="
                    font-size: 28px;
                    font-weight: bold;
                    color: {color};
                    margin-top: 8px;
                ">{prediction}</div>
                <div style="color: #666; margin-top: 4px;">Sentiment</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with result_col2:
        st.markdown("#### Review Summary")
        st.write(f"**Review:** {review_text[:200]}{'…' if len(review_text) > 200 else ''}")
        st.write(f"**Cost:** ₹{cost}  |  **Cuisine:** {cuisine}  |  **Photos:** {pictures}")
        st.write(f"**Word count:** {len(review_text.split())}")

        # Cleaned preview
        clean = preprocess_review(review_text)
        with st.expander("🔍 Show preprocessed text"):
            st.text(clean if clean else "(empty after cleaning)")

    # ── Confidence Chart ──────────────────────────────────────────────────────
    if confidence_data:
        st.markdown("#### Confidence Scores")
        conf_df = pd.DataFrame.from_dict(
            confidence_data, orient="index", columns=["Confidence"]
        ).sort_values("Confidence", ascending=True)

        colors_map = {"Positive": "#28a745", "Neutral": "#ffc107", "Negative": "#dc3545"}
        bar_colors = [colors_map.get(cls, "#666") for cls in conf_df.index]

        st.bar_chart(conf_df, color=bar_colors[0] if len(set(bar_colors)) == 1 else "#0066cc")
        for cls, conf in sorted(confidence_data.items(), key=lambda x: -x[1]):
            st.write(f"- **{cls}**: {conf*100:.1f}%")

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.info(
        "⚠️ **Disclaimer:** This prediction is generated by a machine learning model "
        "trained on historical Zomato review data. It is an automated estimate and may not "
        "perfectly reflect human judgment. Always consider the full context of a review."
    )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#888; font-size:12px;">
        🍽️ Zomato Customer Sentiment Prediction System &nbsp;|&nbsp;
        Built with Scikit-learn & Streamlit &nbsp;|&nbsp;
        Zomato ML Project
    </div>
    """,
    unsafe_allow_html=True,
)
