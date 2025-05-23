from flask import Blueprint, request, jsonify, render_template
from flask import jsonify
from .db import get_db
from .utils import (
    extract_top_phrases_from_reviews,
    fetch_responses_from_db,
    get_aspect_sentiments,
    predict_sentiments,
    get_feedbacks_for_bulk_analysis,
)
from app.utils import get_aspect_sentiments
import pickle
import joblib

bp = Blueprint("routes", __name__)


######## Load Models ########


# Load Models
model = joblib.load("app/models/sentiment_model.pkl")
vectorizer = joblib.load("app/models/vectorizer.pkl")

with open("app/models/ngram_vectorizer.pkl", "rb") as f:
    ngram_vectorizer = pickle.load(f)

with open("app/models/aspects.pkl", "rb") as f:
    aspects = pickle.load(f)


# If you want to use/update the aspects dictionary directly in this file, you can define it here instead of loading from a pickle file.

# aspects = {
#     "food": [
#         "food",
#         "meal",
#         "dish",
#         "taste",
#         "flavor",
#         "spice",
#         "cuisine",
#         "portion",
#         "presentation",
#     ],
#     "service": ["waiter", "staff", "service", "manager", "rude", "polite", "attentive"],
#     "price": [
#         "price",
#         "expensive",
#         "cheap",
#         "cost",
#         "value",
#         "affordable",
#         "overpriced",
#     ],
#     "ambience": [
#         "ambience",
#         "atmosphere",
#         "music",
#         "decor",
#         "environment",
#         "vibe",
#         "lighting",
#     ],
#     "cleanliness": ["clean", "dirty", "hygiene", "sanitary", "neat", "smelly"],
#     "location": ["location", "area", "parking", "nearby", "reachable"],
#     "drinks": ["drink", "juice", "beverage", "wine", "coffee", "tea"],
#     "timeliness": ["wait", "late", "delay", "time", "slow", "fast"],
# }


######## Routes ########


@bp.route("/")
def home():
    return jsonify({"message": "Welcome to the Feedback Analysis API!"})


@bp.route("/chart")
def chart():
    feedbacks = fetch_responses_from_db()
    sentiment_data = predict_sentiments(feedbacks)
    if not sentiment_data:
        return jsonify({"error": "No feedback data available for analysis"}), 400
    # print(sentiment_data)
    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
    for entry in sentiment_data:
        sentiment = entry["sentiment"]
        if sentiment in sentiment_counts:
            sentiment_counts[sentiment] += 1

    return render_template("chart.html", counts=sentiment_counts)


# Route to test database connection
@bp.route("/test_db")
def test_db():
    db = get_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute(
        "SELECT r.response FROM responsend r JOIN  responsendenddata rd ON r.respondentID = rd.id ORDER BY rd.id, r.created_at;"
    )
    data = cursor.fetchall()
    return jsonify(data)


# Route to analyze sentiment for a single feedback
@bp.route("/analyze", methods=["POST"])
def analyze_sentiment():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    return jsonify({"text": text, "sentiment": prediction})


# New route to analyze sentiment for bulk feedbacks
@bp.route("/analyze_bulk", methods=["POST"])
def analyze_bulk():
    """
    Analyze sentiment for bulk feedbacks.
    JSON payload may contain 'feedbacks' list.
    If empty or missing, auto-fetch feedback from DB.
    """
    data = request.get_json(force=True) or {}
    feedbacks = data.get("feedbacks")

    # Get feedbacks from payload or fallback to DB
    feedbacks_to_analyze = get_feedbacks_for_bulk_analysis(feedbacks)

    if not feedbacks_to_analyze:
        return jsonify({"error": "No feedback data available for analysis"}), 400

    results = predict_sentiments(feedbacks_to_analyze)
    return jsonify({"results": results})

    # # New route to fetch responses from DB and analyze them internally
    # @bp.route("/analyze-db-feedback", methods=["GET"])
    # def analyze_db_feedback():
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
    feedbacks = [row["response"] for row in rows if row["response"]]

    if not feedbacks:
        return jsonify({"error": "No feedback responses found in DB"}), 404

    # Use your existing vectorizer and model to predict sentiment for all feedbacks
    vectorized = vectorizer.transform(feedbacks)
    predictions = model.predict(vectorized)

    results = [{"text": f, "sentiment": p} for f, p in zip(feedbacks, predictions)]
    return jsonify({"results": results})


# Frequent Phrases in Feedback Extraction
@bp.route("/frequent_phrases", methods=["POST"])
def extract_frequent_phrases():
    reviews = fetch_responses_from_db()
    if not reviews:
        return jsonify({"error": "No reviews found in DB"}), 400

    top_phrases = extract_top_phrases_from_reviews(reviews)
    if not top_phrases:
        return (
            jsonify(
                {"error": "Could not extract phrases (possibly due to stop words only)"}
            ),
            500,
        )

    return jsonify({"top_phrases": top_phrases})


# Aspect-based Sentiment Analysis
@bp.route("/aspect_analysis", methods=["POST"])
def aspect_analysis():
    """Perform aspect-based sentiment analysis on reviews."""
    reviews = get_feedbacks_for_bulk_analysis()

    if not isinstance(reviews, list) or not all(
        isinstance(r, str) and r.strip() for r in reviews
    ):
        return jsonify({"error": "Expected a list of non-empty review strings"}), 400

    combined_scores = {}
    for review in reviews:
        sentiments = get_aspect_sentiments(review, aspects)
        for aspect, score in sentiments.items():
            combined_scores.setdefault(aspect, []).append(score)

    avg_scores = {
        aspect: round(sum(scores) / len(scores), 2)
        for aspect, scores in combined_scores.items()
        if scores
    }

    star_ratings = {
        aspect: round(((score + 1) / 2) * 4 + 1, 1)
        for aspect, score in avg_scores.items()
    }

    return jsonify(star_ratings)
