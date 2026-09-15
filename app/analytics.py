"""Aggregate sentiment results and pull out simple themes.

Theme extraction is deliberately lightweight for the MVP: stopword-filtered
term frequency split across negative (1-2 star) and positive (4-5 star) reviews.
"""

import re
from collections import Counter

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "so", "of", "to", "in",
    "on", "for", "with", "at", "by", "from", "as", "is", "are", "was", "were",
    "be", "been", "being", "it", "its", "this", "that", "these", "those", "i",
    "me", "my", "we", "our", "you", "your", "he", "she", "they", "them", "their",
    "not", "no", "very", "just", "too", "also", "have", "has", "had", "do",
    "does", "did", "will", "would", "can", "could", "should", "get", "got",
    "one", "two", "after", "before", "when", "while", "than", "there", "here",
    "all", "some", "any", "more", "most", "other", "into", "out", "up", "down",
    "about", "over", "under", "again", "only", "own", "same", "s", "t", "im",
    "ive", "dont", "didnt", "doesnt", "wasnt", "isnt", "id", "item", "product",
    "amazon", "buy", "bought", "purchase", "purchased", "really", "even",
}

_TOKEN_RE = re.compile(r"[a-z]{3,}")


def _themes(texts: list[str], top_n: int = 8) -> list[dict]:
    counts: Counter = Counter()
    for t in texts:
        counts.update(w for w in _TOKEN_RE.findall(t.lower()) if w not in _STOPWORDS)
    return [{"term": w, "count": c} for w, c in counts.most_common(top_n)]


def aggregate(reviews: list[str], predictions: list[dict]) -> dict:
    n = len(predictions)
    dist = {str(s): 0 for s in range(1, 6)}
    for p in predictions:
        dist[str(p["stars"])] += 1

    avg = sum(p["stars"] for p in predictions) / n if n else 0.0
    negative = [r for r, p in zip(reviews, predictions) if p["stars"] <= 2]
    positive = [r for r, p in zip(reviews, predictions) if p["stars"] >= 4]

    return {
        "count": n,
        "average_stars": round(avg, 2),
        "distribution": dist,
        "negative_share": round(len(negative) / n, 4) if n else 0.0,
        "positive_share": round(len(positive) / n, 4) if n else 0.0,
        "negative_themes": _themes(negative),
        "positive_themes": _themes(positive),
    }
