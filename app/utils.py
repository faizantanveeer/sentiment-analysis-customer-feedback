from textblob import TextBlob
from .db import get_db
import pickle
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from .db import get_db
import re


# Load Models
model = joblib.load("app/models/sentiment_model.pkl")
vectorizer = joblib.load("app/models/vectorizer.pkl")

with open("app/models/ngram_vectorizer.pkl", "rb") as f:
    ngram_vectorizer = pickle.load(f)

with open("app/models/aspects.pkl", "rb") as f:
    aspects = pickle.load(f)


def clean_texts(texts):
    return [
        re.sub(r"\s+", " ", t.strip())
        for t in texts
        if isinstance(t, str) and t.strip()
    ]


def extract_top_phrases_from_reviews(
    reviews, ngram_range=(2, 3), max_features=1000, top_n=15
):
    if not reviews:
        return []

    vec = CountVectorizer(
        ngram_range=ngram_range, stop_words="english", max_features=max_features
    )
    try:
        X = vec.fit_transform(reviews)
    except ValueError:
        return []

    sum_words = X.sum(axis=0)
    phrases = sorted(
        [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()],
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]

    sum_words = X.sum(axis=0)
    phrases = sorted(
        [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()],
        key=lambda x: x[1],
        reverse=True,
    )

    # Remove phrases that are substrings of longer, more frequent phrases
    unique_phrases = []
    seen = set()
    for phrase, count in phrases:
        if any(phrase in longer for longer, _ in unique_phrases):
            continue
        # Filter out phrases that are too short or not meaningful
        if len(phrase.split()) < 2:
            continue
        if phrase in seen:
            continue
        unique_phrases.append((phrase, int(count)))
        seen.add(phrase)
        if len(unique_phrases) >= top_n:
            break

    return unique_phrases


def fetch_responses_from_db():
    """Fetch all responses from the database, ordered."""
    db = get_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute(
        """
        SELECT r.response 
        FROM responsend r 
        JOIN responsendenddata rd ON r.respondentID = rd.id 
        ORDER BY rd.id, r.created_at;
    """
    )
    rows = cursor.fetchall()
    return [
        row["response"] for row in rows if row["response"] and row["response"].strip()
    ]


def get_aspect_sentiments(review, aspects):
    if not review.strip():
        return {}

    blob = TextBlob(review)  # ✅ Make sure this is always defined

    aspect_scores = {aspect: [] for aspect in aspects}

    for sentence in blob.sentences:
        lower_sentence = sentence.lower()
        for aspect, keywords in aspects.items():
            if any(keyword in lower_sentence for keyword in keywords):
                aspect_scores[aspect].append(sentence.sentiment.polarity)

    return {
        aspect: round(sum(scores) / len(scores), 2)
        for aspect, scores in aspect_scores.items()
        if scores
    }


def predict_sentiments(texts):
    """Predict sentiment labels for a list of texts."""
    if not texts:
        return []
    vectorized = vectorizer.transform(texts)
    predictions = model.predict(vectorized)
    return [{"text": t, "sentiment": p} for t, p in zip(texts, predictions)]


def get_feedbacks_for_bulk_analysis(feedbacks=None):
    """
    Return feedback list to analyze.
    If feedbacks is provided and non-empty list, use it.
    Otherwise, fetch feedbacks from DB.
    """
    if (
        feedbacks
        and isinstance(feedbacks, list)
        and any(f.strip() for f in feedbacks if isinstance(f, str))
    ):
        # Clean list: keep only non-empty strings
        clean_feedbacks = [
            f.strip() for f in feedbacks if isinstance(f, str) and f.strip()
        ]
        return clean_feedbacks
    # fallback to DB fetch
    return fetch_responses_from_db()


def validate_json_keys(json_data, required_keys):
    """Validate that required keys exist in JSON payload."""
    missing_keys = [key for key in required_keys if key not in json_data]
    if missing_keys:
        return False, f"Missing keys: {', '.join(missing_keys)}"
    return True, None
