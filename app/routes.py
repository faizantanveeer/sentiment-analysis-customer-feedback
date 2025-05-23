from flask import Blueprint, request, jsonify
from .db import get_db
from .utils import (
    extract_top_phrases_from_reviews,
    fetch_responses_from_db,
    get_aspect_sentiments,
    predict_sentiments,
    get_feedbacks_for_bulk_analysis,
    validate_json_keys,
    clean_texts,
)
from app.utils import get_aspect_sentiments
import pickle
import joblib

bp = Blueprint("routes", __name__)


# Load Models
model = joblib.load("app/models/sentiment_model.pkl")
vectorizer = joblib.load("app/models/vectorizer.pkl")

with open("app/models/ngram_vectorizer.pkl", "rb") as f:
    ngram_vectorizer = pickle.load(f)

with open("app/models/aspects.pkl", "rb") as f:
    aspects = pickle.load(f)

# Existing routes...


@bp.route("/")
def home():
    return jsonify({"message": "Welcome to the Feedback Analysis API!"})


@bp.route("/test-db")
def test_db():
    db = get_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute(
        "SELECT r.response FROM responsend r JOIN  responsendenddata rd ON r.respondentID = rd.id ORDER BY rd.id, r.created_at;"
    )
    data = cursor.fetchall()
    return jsonify(data)


@bp.route("/analyze", methods=["POST"])
def analyze_sentiment():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    return jsonify({"text": text, "sentiment": prediction})


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


# New route to fetch responses from DB and analyze them internally
@bp.route("/analyze-db-feedback", methods=["GET"])
def analyze_db_feedback():
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


@bp.route("/aspect-analysis", methods=["POST"])
def aspect_analysis():
    """Perform aspect-based sentiment analysis on reviews."""
    data = get_feedbacks_for_bulk_analysis()

    valid, error = validate_json_keys(data, ["response"])
    if not valid:
        return jsonify({"error": error}), 400

    reviews = data["reviews"]
    if not isinstance(reviews, list) or not all(
        isinstance(r, str) and r.strip() for r in reviews
    ):
        return jsonify({"error": "Reviews must be a list of non-empty strings"}), 400

    combined = {}
    for review in reviews:
        sentiments = get_aspect_sentiments(review, aspects)
        for aspect, score in sentiments.items():
            combined.setdefault(aspect, []).append(score)

    avg_scores = {k: round(sum(v) / len(v), 2) for k, v in combined.items()}
    # Convert from [-1,1] sentiment score to 1-5 star rating scale
    star_ratings = {k: round(((v + 1) / 2) * 4 + 1, 1) for k, v in avg_scores.items()}
    return jsonify(star_ratings)
