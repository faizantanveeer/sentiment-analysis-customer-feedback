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
from collections import defaultdict
from datetime import datetime
from collections import Counter


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


@bp.route("/chart", methods=["POST"])
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

    return jsonify(sentiment_counts)

    # @bp.route("/client_chart", methods=["POST"])
    # def client_chart():
    data = request.get_json(force=True)
    client_id = data.get("client_id")

    if not client_id:
        return jsonify({"error": "client_id is required"}), 400

    db = get_db()
    cursor = db.cursor(dictionary=True)

    query = """
    SELECT r.*
    FROM responsend r
    JOIN locations l ON r.locationID = l.ID
    JOIN hieararchylevels h ON l.hiearchylevelID = h.ID
    JOIN (
        SELECT f.client_id, MAX(hh.level) AS max_level
        FROM formats f
        JOIN hieararchylevels hh ON f.assignHiearchy = hh.hiearchyid
        WHERE f.client_id = %(client_id)s
        GROUP BY f.client_id
    ) max_h ON h.level = max_h.max_level AND max_h.client_id = %(client_id)s
    JOIN formats f ON f.assignHiearchy = h.hiearchyid
    JOIN users u ON u.id = f.client_id
    WHERE u.role_id = 2 AND f.client_id = %(client_id)s;

    """
    cursor.execute(query, {"client_id": client_id})

    rows = cursor.fetchall()

    if not rows:
        return jsonify({"error": "No responses found for the given client"}), 404

    feedbacks = [row["response"] for row in rows if row["response"]]

    if not feedbacks:
        return jsonify({"error": "No valid responses found"}), 404

    # Predict sentiments
    vectorized = vectorizer.transform(feedbacks)
    predictions = model.predict(vectorized)

    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
    for sentiment in predictions:
        if sentiment in sentiment_counts:
            sentiment_counts[sentiment] += 1
    return jsonify(sentiment_counts)


@bp.route("/client_chart", methods=["POST"])
def client_chart():
    data = request.get_json(force=True)
    client_id = data.get("client_id")
    location_id = data.get("location_id")  # optional

    print("client_id:", client_id)

    # if not client_id:
    #     return jsonify({"error": "client_id is required"}), 400

    db = get_db()
    cursor = db.cursor(dictionary=True)

    query = """
    SELECT r.response,
           DATE(r.Date) AS created_date
    FROM responsend r
    JOIN locations l ON r.locationID = l.ID
    JOIN hieararchylevels h ON l.hiearchylevelID = h.ID
    JOIN (
        SELECT f.client_id, MAX(hh.level) AS max_level
        FROM formats f
        JOIN hieararchylevels hh ON f.assignHiearchy = hh.hiearchyid
        WHERE (%(client_id)s IS NULL OR f.client_id = %(client_id)s)
        GROUP BY f.client_id
    ) max_h ON h.level = max_h.max_level 
           AND (%(client_id)s IS NULL OR max_h.client_id = %(client_id)s)
    JOIN formats f ON f.assignHiearchy = h.hiearchyid
    JOIN users u ON u.id = f.client_id
    WHERE u.role_id = 2 
      AND (%(client_id)s IS NULL OR f.client_id = %(client_id)s)
    {location_filter};
    """

    params = {"client_id": client_id}
    location_filter = ""

    if location_id:
        location_filter = "AND r.locationID = %(location_id)s"
        params["location_id"] = location_id

    query = query.format(location_filter=location_filter)
    cursor.execute(query, params)
    rows = cursor.fetchall()

    print("Fetched rows:", len(rows))

    if not rows:
        return jsonify({"error": "No responses found"}), 404

    feedbacks = [
        {"response": row["response"], "date": row["created_date"]}
        for row in rows
        if row["response"]
    ]

    if not feedbacks:
        return jsonify({"error": "No valid responses"}), 404

    # --- Sentiment Analysis ---
    response_texts = [item["response"] for item in feedbacks]

    dates = [item["date"] for item in feedbacks]

    vectorized = vectorizer.transform(response_texts)
    predictions = model.predict(vectorized)

    # Map numeric labels to readable ones
    # label_map = {0: "negative", 1: "neutral", 2: "positive"}
    sentiments = predictions.tolist()

    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
    for sent in sentiments:
        sentiment_counts[sent] += 1

    sentiment_chart = {
        "title": "Sentiment Distribution",
        "type": "bar",
        "labels": list(sentiment_counts.keys()),
        "data": list(sentiment_counts.values()),
    }

    # --- Aspect-Based Ratings ---
    combined_scores = {}
    for review in response_texts:
        sentiments_map = get_aspect_sentiments(review, aspects)
        for aspect, score in sentiments_map.items():
            combined_scores.setdefault(aspect, []).append(score)

    def map_to_5_star(score):
        return round(((score + 1) / 2) * 4 + 1, 2)

    avg_scores = {
        aspect: map_to_5_star(sum(scores) / len(scores))
        for aspect, scores in combined_scores.items()
        if scores
    }

    aspect_chart = {
        "title": "Aspect-Based Ratings",
        "type": "polarArea",  # or 'radar'
        "labels": list(avg_scores.keys()),
        "data": list(avg_scores.values()),
    }

    # --- Frequent Phrases ---
    top_phrases = extract_top_phrases_from_reviews(response_texts)
    phrases_chart = {
        "title": "Most Frequent Phrases",
        "type": "bar",
        "labels": [p[0] for p in top_phrases],
        "data": [p[1] for p in top_phrases],
        "options": {"indexAxis": "y"},
    }

    # --- Sentiment Trend Chart ---
    # Build: { "YYYY-MM": {"positive": 2, "neutral": 1, "negative": 0} }

    sentiment_counts = Counter(predictions)
    total = len(predictions)

    sentiment_percentages = {
        sentiment: round((count / total) * 100, 2)
        for sentiment, count in sentiment_counts.items()
    }

    sentiment_dist_chart = {
        "title": "Sentiment Distribution",
        "type": "pie",
        "labels": list(sentiment_percentages.keys()),
        "data": list(sentiment_percentages.values()),
        "options": {
            "scales": {
                "y": {
                    "beginAtZero": True,
                    "max": 100,
                    "title": {"display": True, "text": "Percentage (%)"},
                }
            }
        },
    }

    # Count sentiments per month
    monthly_counts = defaultdict(lambda: {"positive": 0, "neutral": 0, "negative": 0})

    for i, sentiment in enumerate(predictions):
        date_obj = dates[i]
        if not isinstance(date_obj, datetime):
            date_obj = datetime.strptime(str(date_obj), "%Y-%m-%d")
        key = date_obj.strftime("%b-%Y")  # e.g., Apr-2025
        monthly_counts[key][sentiment] += 1

    # Sort months chronologically
    sorted_months = sorted(monthly_counts.keys())

    # Convert to percentages
    trend_data = {"positive": [], "neutral": [], "negative": []}
    for month in sorted_months:
        total = sum(monthly_counts[month].values())
        for sentiment in trend_data:
            percent = (
                round((monthly_counts[month][sentiment] / total) * 100, 2)
                if total
                else 0
            )
            trend_data[sentiment].append(percent)

    # print("Monthly Sentiment Trend (%):")
    # for i, month in enumerate(sorted_months):
    #     print(f"{month}:")
    #     for sentiment in trend_data:
    #         print(f"  {sentiment}: {trend_data[sentiment][i]}%")

    trend_chart = {
        "title": "Sentiment Trend Over Time",
        "type": "line",
        "labels": sorted_months,
        "datasets": [
            {
                "label": "Positive",
                "data": trend_data["positive"],
                "borderColor": "green",
                "fill": False,
                "tension": 0.3,
            },
            {
                "label": "Neutral",
                "data": trend_data["neutral"],
                "borderColor": "orange",
                "fill": False,
                "tension": 0.3,
            },
            {
                "label": "Negative",
                "data": trend_data["negative"],
                "borderColor": "red",
                "fill": False,
                "tension": 0.3,
            },
        ],
    }

    return jsonify(
        {
            "charts": [
                sentiment_chart,
                aspect_chart,
                phrases_chart,
                trend_chart,
                sentiment_dist_chart,
            ]
        }
    )


# Route to test database connection
@bp.route("/test_db")
def test_db():
    db = get_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute(
        "SELECT r.response, Date(r.Date) as date FROM responsend r JOIN  responsendenddata rd ON r.respondentID = rd.id ORDER BY rd.id, r.created_at;"
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
