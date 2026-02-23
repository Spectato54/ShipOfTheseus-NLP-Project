"""
Cleans raw text and generates text stats.
"""

import re


def clean_text(text):
    """
    Clean the input text by removing extra whitespace and normalizing it.
    """
    sentences = text.split(" ")
    cleaned_sentences = []
    for s in sentences:
        s = s.lower()  # Normalize to lowercase
        s = re.sub(r"[^a-zA-Z0-9]", "", s)  # Remove non-alphanumeric chars
        s = s.strip()  # Remove leading/trailing whitespace
        if s:  # Only keep non-empty sentences
            cleaned_sentences.append(s)
    return " ".join(cleaned_sentences)


def remove_stopwords(text, stopwords):
    """
    Remove stopwords from the input text.
    """
    sentences = text.split(" ")
    filtered_sentences = [s for s in sentences if s not in stopwords]
    return " ".join(filtered_sentences)


def preprocess_dataset(df, stopwords):
    """
    Preprocess the dataset by cleaning text and removing stopwords.
    Returns a new DataFrame with cleaned text and added 'cleaned_text', 'family', 'no_stop_text' columns.
    """
    df = df.copy()
    df["cleaned_text"] = df["text"].apply(clean_text)
    df["no_stop_text"] = df["cleaned_text"].apply(
        lambda x: remove_stopwords(x, stopwords)
    )
    return df


def text_stats(df):
    """
    Generate text statistics for the dataset.
    Returns a DataFrame with average length of original, cleaned, and no-stopword text.
    """
    stats = df.copy()
    word_count = stats["text"].apply(lambda x: len(x.split()))
    cleaned_word_count = stats["cleaned_text"].apply(lambda x: len(x.split()))
    no_stop_word_count = stats["no_stop_text"].apply(lambda x: len(x.split()))
    unique_words = stats["cleaned_text"].apply(lambda x: len(set(x.split())))
    stats["word_count"] = word_count
    stats["cleaned_word_count"] = cleaned_word_count
    stats["no_stop_word_count"] = no_stop_word_count
    stats["unique_words"] = unique_words
    return stats[
        ["word_count", "cleaned_word_count", "no_stop_word_count", "unique_words"]
    ]
