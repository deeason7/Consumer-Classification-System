# src/preprocessing/transformer.py
"""
Provides functions for feature engineering on the cleaned consumer complaint data.
It includes sentiment analysis, target encoding, and the creation of interaction and text-based features like topic models.
"""

import pandas as pd
import numpy as np
import re
from textblob import TextBlob
from nltk import pos_tag
import nltk
from typing import List, Dict, Tuple, Union

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Keywords Specific to Financial Complaints
extreme_negative_keywords = [
    "fraud", "harassment", "lawsuit", "illegal", "identity theft", "repossession",
    "foreclosure", "stolen", "unauthorized", "charge-off", "eviction", "bankruptcy"
]

negative_keywords = [
    "delay", "denied", "slow", "error", "dispute", "complaint", "billing", "fees", "late fee",
    "overcharged", "wrong", "issue", "problem", "scam", "unresponsive", "ignored", "collections"
]

neutral_indicators = [
    "opened account", "submitted form", "changed address", "received notice", "updated info"
]

emphasis_words = [
    "very", "extremely", "completely", "utterly", "absolutely", "seriously", "highly"
]

negation_words = [
    "not", "never", "no", "none", "nobody", "nothing", "nowhere", "hardly", "scarcely", "barely"
]

# Runtime NLTK downloads if missing
def safe_nltk_download(resource_name):
    """
    Ensure that the given NLTK resource is available locally; download if it's missing.

    Args:
        resource_name (str): The name of the NLTK resource to check/download.
    """
    from nltk.data import find
    try:
        find(resource_name)
    except LookupError:
        nltk.download(resource_name)

# Ensure required tokenizers and taggers are available
safe_nltk_download("punkt")
safe_nltk_download("averaged_perceptron_tagger")

def basic_tokenize(text: str):
    """
    Tokenize text into words by removing punctuation and splitting on whitespace.

    Args:
        text (str): The input text to tokenize.

    Returns:
        List[str]: List of word tokens.
    """
     # Remove all punctuation characters
    text = re.sub(r"[^\w\s]", "", text)
    # Split on whitespace to generate tokens
    return text.split()

# Weak Sentiment Classifier
def weak_sentiment(text: str) -> Tuple[str, float]:
    """
    Perform a simple, rule-based sentiment classification on text into
    'neutral', 'negative', or 'extreme_negative' and intensity scoring.

    Uses keyword presence, TextBlob polarity, and part-of-speech tags to adjust scoring.

    Args:
        text (str): The input complaint text.

    Returns:
        Tuple[str, float]: A tuple containing the sentiment label and the polarity score.
    """

    # Handle non-string or very short inputs
    if not isinstance(text, str) or len(text.strip()) < 10:
        return "neutral", 0.0

    text_lower = text.lower()
    words = basic_tokenize(text_lower)

    #POS tagging; fallback to empty
    try:
        tokens_pos = pos_tag(words)
    except Exception:
        tokens_pos = []

    # Compute TextBlob sentiment metrics
    blob = TextBlob(text_lower)
    polarity = blob.sentiment.polarity

    # Keyword Signals Checks
    extreme_hit = any(word in text_lower for word in extreme_negative_keywords)
    negative_hit = any(word in text_lower for word in negative_keywords)
    neutral_hit = any(phrase in text_lower for phrase in neutral_indicators)

    # Count emphasis words in text
    emphasis_count = sum(1 for word, _ in tokens_pos if word in emphasis_words)
    
    #Detect presence of negation terms
    negation_detected = any(word in words for word in negation_words)

    # Cumulative score from polarity
    cumulative_score = polarity
    
    # Penalize for extreme and negative keyword hits
    cumulative_score -= 0.3 * int(extreme_hit)
    cumulative_score -= 0.2 * int(negative_hit)
    
    #Reward for emphasis words
    cumulative_score += 0.1 * emphasis_count

    #Invert score if negation words detected
    if negation_detected:
        cumulative_score *= -0.7

    # Final Rules
    label = "neutral" # Default
    if extreme_hit:
        label = "extreme_negative"
    if cumulative_score < -0.2 or negative_hit:
        label = "negative"
    if neutral_hit and polarity > -0.1:
        label = "neutral"
    if polarity > 0:
        #Positive polarity is treated as neutral in this context
        label = "neutral"

    return label, polarity

# Topic Modeling Feature Creation
def add_topic_feature(df: pd.DataFrame, n_topics: int = 5) -> pd.DataFrame:
    """
    Adds topic modeling features to the DataFrame using TF-IDF and LDA

    Args:
        df (pd.DataFrame): The DataFrame with a 'text_cleaned' column.
        n_topics (int): The number of topics to generate.

    Returns:
        pd.DataFrame: The DataFrame with added 'topic_X' columns
    """
    # Vectorize
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1,2))
    tfidf_matrix = vectorizer.fit_transform(df['text_cleaned'])

    # Apply Latent Dirichlet Allocation
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    topic_dist = lda.fit_transform(tfidf_matrix)

    # Add topic distribution as feature to the DataFrame
    for i in range(n_topics):
        df[f'topic_{i}'] = topic_dist[:, i]

    return df

# Target Encoding
def calculate_target_encoding(df: pd.DataFrame, group_col: str, target_col: str, min_samples_leaf=20, smoothing=10) -> dict:
    """
    Compute target-encoded values for categories in group_col with smoothing.

    Args:
        df (pd.DataFrame): DataFrame containing data.
        group_col (str): Column name of categorical variable to encode.
        target_col (str): Column name of binary target variable.
        min_samples_leaf (int): Minimum samples to take category average into account.
        smoothing (float): Smoothing effect to balance category vs global mean.

    Returns:
        dict: Mapping from category to its smoothed mean target value.
    """
    #Calculate mean and count per category
    averages = df.groupby(group_col)[target_col].agg(["mean", "count"])
    
    #Smoothing factor based on counts
    smoothing_factor = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    global_mean = df[target_col].mean()
    
    #Compute smoothed means blending global and category mean
    averages["smoothed_mean"] = global_mean * (1 - smoothing_factor) + averages["mean"] * smoothing_factor
    return averages["smoothed_mean"].to_dict()

# Feature Transformer
def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform full feature engineering pipeline for complaint dispute modeling,
    adding binary flags, encodings, sentiment labels, topic models and interaction terms.

    Args:
        df (pd.DataFrame): Raw complaints DataFrame with required columns.

    Returns:
        pd.DataFrame: Transformed DataFrame ready for modeling.
    """
    df = df.copy()

    # Binary Encoding
    df["consumer_disputed_binary"] = df["consumer_disputed?"].map({"Yes": 1, "No": 0})
    df["timely_response_binary"] = df["timely_response"].map({"Yes": 1, "No": 0})

    # Handle missing states
    df["state"] = df["state"].fillna("Unknown")

    # Text Features: Length if cleaned narratives
    df["text_length"] = df["text_cleaned"].apply(lambda x: len(x.split()))

    # Compute and map target-encoded dispute rates for product and company
    product_encoding = calculate_target_encoding(df, "product", "consumer_disputed_binary")
    company_encoding = calculate_target_encoding(df, "company", "consumer_disputed_binary")
    df["product_dispute_rate"] = df["product"].map(product_encoding).fillna(df["consumer_disputed_binary"].mean())
    df["company_dispute_rate"] = df["company"].map(company_encoding).fillna(df["consumer_disputed_binary"].mean())

    # Apply weak sentiment classifier to get both label and sentiment
    sentiment_results = df["text_cleaned"].apply(weak_sentiment)
    df["sentiment"] = sentiment_results.apply(lambda x: x[0])
    df["sentiment_intensity"] = sentiment_results.apply(lambda  x: x[1])

    df["sentiment_encoded"] = df["sentiment"].map({"neutral": 0, "negative": 1, "extreme_negative": 2}).fillna(0).astype(int)

    # Interaction features combining sentiment/response and company/response
    df["sentiment_timely_interaction"] = df["sentiment_encoded"] * df["timely_response_binary"]
    df["company_timely_interaction"] = df["company_dispute_rate"] * df["timely_response_binary"]

    # Drop Irrelevant Columns
    drop_cols = [
        "tags", "company_public_response", "consumer_consent_provided", "consumer_complaint_narrative",
        "complaint_id", "date_received", "date_sent_to_company", "submitted_via", "zipcode",
        "sub_issue", "sub_product", "consumer_disputed?", "company_response_to_consumer"
    ]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")

    return df