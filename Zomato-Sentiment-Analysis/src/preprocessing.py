"""
preprocessing.py
----------------
NLP text preprocessing for Zomato review text.
"""

import os
import re
import string
import warnings
import pandas as pd
import nltk

warnings.filterwarnings("ignore")

# ─── Configure local NLTK data paths ──────────────────────────────────────────
for candidate in [
    os.path.join(os.path.dirname(__file__), "..", "nltk_data"),
    os.path.join(os.path.dirname(__file__), "..", "..", "nltk_data"),
    os.path.join(os.getcwd(), "nltk_data"),
    os.path.join(os.getcwd(), "Zomato-Sentiment-Analysis", "nltk_data"),
]:
    abs_cand = os.path.abspath(candidate)
    if os.path.exists(abs_cand) and abs_cand not in nltk.data.path:
        nltk.data.path.insert(0, abs_cand)

# ─── Ensure assets are loaded ─────────────────────────────────────────────────
# Check common corpus paths without attempting network download in sandbox
for resource_path in ["corpora/stopwords", "tokenizers/punkt", "corpora/wordnet",
                     "taggers/averaged_perceptron_tagger", "corpora/omw-1.4", "tokenizers/punkt_tab"]:
    try:
        nltk.data.find(resource_path)
    except LookupError:
        pass

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

STOP_WORDS = set(stopwords.words("english"))
# Preserve negation words — they are critical for sentiment
NEGATION_WORDS = {"not", "no", "never", "nor", "neither", "without", "barely",
                  "hardly", "scarcely", "seldom", "nothing", "nobody", "none"}
STOP_WORDS -= NEGATION_WORDS

LEMMATIZER = WordNetLemmatizer()

# Contraction map (common English contractions)
CONTRACTIONS = {
    "won't": "will not", "can't": "cannot", "n't": " not",
    "'re": " are", "'s": " is", "'d": " would", "'ll": " will",
    "'t": " not", "'ve": " have", "'m": " am",
    "i'm": "i am", "you're": "you are", "he's": "he is",
    "she's": "she is", "it's": "it is", "we're": "we are",
    "they're": "they are", "i've": "i have", "you've": "you have",
    "we've": "we have", "they've": "they have", "i'd": "i would",
    "you'd": "you would", "he'd": "he would", "she'd": "she would",
    "we'd": "we would", "they'd": "they would", "i'll": "i will",
    "you'll": "you will", "he'll": "he will", "she'll": "she will",
    "we'll": "we will", "they'll": "they will", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "hasn't": "has not", "haven't": "have not", "hadn't": "had not",
    "doesn't": "does not", "don't": "do not", "didn't": "did not",
    "wouldn't": "would not", "couldn't": "could not", "shouldn't": "should not",
    "mightn't": "might not", "mustn't": "must not",
}


def expand_contractions(text: str) -> str:
    """Expand common English contractions."""
    for contraction, expansion in CONTRACTIONS.items():
        text = re.sub(re.escape(contraction), expansion, text, flags=re.IGNORECASE)
    return text


def remove_html_tags(text: str) -> str:
    """Strip HTML tags if any."""
    return re.sub(r"<[^>]+>", " ", text)


def remove_urls(text: str) -> str:
    """Remove http/https URLs."""
    return re.sub(r"http\S+|www\.\S+", " ", text)


def remove_special_characters(text: str) -> str:
    """Keep alphanumeric characters and spaces; remove other special chars."""
    return re.sub(r"[^a-zA-Z0-9\s]", " ", text)


def clean_whitespace(text: str) -> str:
    """Collapse multiple whitespace into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list:
    """Tokenise text into word tokens."""
    return word_tokenize(text)


def remove_stopwords(tokens: list) -> list:
    """Remove stopwords, preserving negation words."""
    return [t for t in tokens if t not in STOP_WORDS]


def lemmatize_tokens(tokens: list) -> list:
    """Lemmatize each token."""
    return [LEMMATIZER.lemmatize(t) for t in tokens]


def preprocess_review(text: str) -> str:
    """
    Full NLP preprocessing pipeline applied to a single review string.
    Steps:
        1. Expand contractions
        2. Remove HTML tags
        3. Remove URLs
        4. Lowercase
        5. Remove special characters (keeps alphanumeric + spaces)
        6. Tokenize
        7. Remove stopwords (preserving negations)
        8. Lemmatize
        9. Rejoin and clean whitespace
    """
    if pd.isna(text) or str(text).strip() == "":
        return ""

    text = str(text)
    text = expand_contractions(text)
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = text.lower()
    text = remove_special_characters(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    text = " ".join(tokens)
    text = clean_whitespace(text)
    return text


def apply_text_preprocessing(df: pd.DataFrame,
                              review_col: str = "Review",
                              output_col: str = "Clean_Review") -> pd.DataFrame:
    """
    Apply preprocess_review to every row in a DataFrame column.
    Returns a new DataFrame with the Clean_Review column added.
    """
    df = df.copy()
    print(f"[NLP] Preprocessing {len(df)} reviews …")
    df[output_col] = df[review_col].apply(preprocess_review)
    empty_after = (df[output_col].str.strip() == "").sum()
    print(f"[NLP] Empty reviews after cleaning : {empty_after}")
    return df


def pos_tag_sample(text: str) -> list:
    """Return POS tags for a sample text (for demonstration in notebook)."""
    tokens = word_tokenize(text)
    return nltk.pos_tag(tokens)


if __name__ == "__main__":
    sample = "It's not a bad place! Won't visit again though. The food wasn't great."
    print("Original  :", sample)
    print("Cleaned   :", preprocess_review(sample))
