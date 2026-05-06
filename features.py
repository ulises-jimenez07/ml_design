"""
Feature preprocessing for the Two-Tower recommendation system.

Implements the 128-dimensional feature vectors described in TUTORIAL.md Section 5.2.
Each block is designed to be swapped out for a production-quality equivalent:

  Hash-based category/country embeddings
      → nn.Embedding tables trained end-to-end with the towers

  Bag-of-words title embedding
      → Vertex AI text-embedding API (textembedding-gecko@003), pre-computed
        and stored in BigQuery, refreshed nightly

  Zero click-sequence placeholder
      → Mean-pooled EventTower embeddings of the last 10 interactions, OR a
        GRU/Transformer session encoder for session-based recommendations

Usage:
    from features import UserFeatureEncoder, EventFeatureEncoder

    user_enc  = UserFeatureEncoder()
    event_enc = EventFeatureEncoder()

    user_vec  = user_enc.encode(row_dict)   # np.ndarray (128,) float32
    event_vec = event_enc.encode(row_dict)  # np.ndarray (128,) float32

Run self-tests:
    python features.py
"""

import hashlib
import math
import numpy as np

# --------------------------------------------------------------------------- #
# Dimension layout — mirrors TUTORIAL.md Section 5.2 exactly                 #
# --------------------------------------------------------------------------- #

# User tower (128 dims total)
USER_AGE_DIMS      = 16   # dims  0–15:  age-bucket one-hot (16 buckets)
USER_GENDER_DIMS   =  3   # dims 16–18:  gender one-hot [M, F, Unknown]
USER_COUNTRY_DIMS  = 32   # dims 19–50:  country pseudo-embedding
USER_CATEGORY_DIMS = 10   # dims 51–60:  top-10 category purchase proportions
USER_ACTIVITY_DIMS = 10   # dims 61–70:  log-scaled activity counters
USER_RECENCY_DIMS  = 10   # dims 71–80:  recency / decay signals
USER_SEQUENCE_DIMS = 47   # dims 81–127: recent click-sequence mean-pool

assert (USER_AGE_DIMS + USER_GENDER_DIMS + USER_COUNTRY_DIMS +
        USER_CATEGORY_DIMS + USER_ACTIVITY_DIMS + USER_RECENCY_DIMS +
        USER_SEQUENCE_DIMS) == 128, "User dim layout must sum to 128"

# Event tower (128 dims total)
EVENT_CATEGORY_DIMS = 32   # dims   0–31: category embedding
EVENT_PRICE_DIMS    =  8   # dims  32–39: price-tier one-hot (8 buckets)
EVENT_POP_DIMS      =  8   # dims  40–47: popularity signals
EVENT_TEMPORAL_DIMS =  8   # dims  48–55: temporal signals
EVENT_VENUE_DIMS    =  8   # dims  56–63: city/venue pseudo-embedding
EVENT_TEXT_DIMS     = 64   # dims  64–127: title/description text embedding

assert (EVENT_CATEGORY_DIMS + EVENT_PRICE_DIMS + EVENT_POP_DIMS +
        EVENT_TEMPORAL_DIMS + EVENT_VENUE_DIMS + EVENT_TEXT_DIMS) == 128, \
    "Event dim layout must sum to 128"

# --------------------------------------------------------------------------- #
# Shared helpers                                                               #
# --------------------------------------------------------------------------- #

def _hash_embedding(value: str, dims: int) -> np.ndarray:
    """
    Deterministic unit-normalised pseudo-embedding seeded from value's SHA-256.

    Assigns a stable point on the unit hypersphere to every unique string so
    the pipeline runs without a vocabulary file or pre-trained weights.

    Production replacement (choose one based on cardinality):
        Low cardinality  (<10k values): nn.Embedding table, trained end-to-end.
        High cardinality (text fields): Vertex AI textembedding-gecko@003,
            pre-computed offline and stored in a BigQuery lookup table.
    """
    seed = int(hashlib.sha256(str(value).encode()).hexdigest(), 16) % (2 ** 32)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dims).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _log_scale(x: float) -> float:
    """log1p of non-negative x; returns 0.0 for missing / negative values."""
    return float(np.log1p(max(float(x), 0.0)))


# --------------------------------------------------------------------------- #
# User feature encoder                                                         #
# --------------------------------------------------------------------------- #

_AGE_BUCKET_WIDTH = 5       # 5-year buckets starting at age 15
_GENDERS = ["M", "F"]       # index 2 = unknown / other / not set

# Top-10 categories from bigquery-public-data.thelook_ecommerce.
# Replace with your platform's top-10 event/product categories.
_TOP_CATEGORIES = [
    "Outerwear & Coats", "Jeans", "Suits & Sport Coats",
    "Sweaters", "Fashion Hoodies & Sweatshirts",
    "Sleep & Lounge", "Dresses", "Tops & Tees",
    "Active", "Intimates & Sleepwear",
]


class UserFeatureEncoder:
    """
    Converts a raw user feature dict to a 128-dim float32 numpy array.

    Input dict keys (all optional — missing values default to 0 / 'Unknown'):
        age                  int    User age in years
        gender               str    'M', 'F', or anything else → unknown bucket
        country              str    ISO-2 country code, e.g. 'US'
        top_category         str    User's most-purchased category
        total_sessions       float  Lifetime session count
        total_events         float  Lifetime event (click/view) count
        active_days          float  Count of distinct active calendar days
        total_purchases      float  Lifetime purchase count
        days_since_last_active float Days since most recent session
        account_age_days     float  Days since account creation
        purchase_rate        float  total_purchases / total_sessions
        recent_item_ids      list   Up to 10 most-recently clicked item IDs
    """

    def encode(self, row: dict) -> np.ndarray:
        parts = [
            self._age_onehot(row.get("age", 0)),
            self._gender_onehot(row.get("gender", "")),
            _hash_embedding(row.get("country", "XX"), USER_COUNTRY_DIMS),
            self._category_proportions(row.get("top_category", "")),
            self._activity_features(row),
            self._recency_features(row),
            self._sequence_embedding(row.get("recent_item_ids", [])),
        ]
        vec = np.concatenate(parts).astype(np.float32)
        assert vec.shape == (128,), f"UserFeatureEncoder: expected (128,), got {vec.shape}"
        return vec

    def _age_onehot(self, age: int) -> np.ndarray:
        vec = np.zeros(USER_AGE_DIMS, dtype=np.float32)
        if age and age > 0:
            bucket = min(int(age) // _AGE_BUCKET_WIDTH - 2, USER_AGE_DIMS - 1)
            vec[max(bucket, 0)] = 1.0
        return vec

    def _gender_onehot(self, gender: str) -> np.ndarray:
        vec = np.zeros(USER_GENDER_DIMS, dtype=np.float32)
        vec[_GENDERS.index(gender) if gender in _GENDERS else 2] = 1.0
        return vec

    def _category_proportions(self, top_category: str) -> np.ndarray:
        vec = np.zeros(USER_CATEGORY_DIMS, dtype=np.float32)
        if top_category in _TOP_CATEGORIES:
            vec[_TOP_CATEGORIES.index(top_category)] = 1.0
        return vec

    def _activity_features(self, row: dict) -> np.ndarray:
        raw = [
            _log_scale(row.get("total_sessions", 0)),
            _log_scale(row.get("total_events", 0)),
            _log_scale(row.get("active_days", 0)),
            _log_scale(row.get("total_purchases", 0)),
            float(np.clip(row.get("purchase_rate", 0.0), 0.0, 1.0)),
        ]
        padded = raw + [0.0] * (USER_ACTIVITY_DIMS - len(raw))
        return np.array(padded[:USER_ACTIVITY_DIMS], dtype=np.float32)

    def _recency_features(self, row: dict) -> np.ndarray:
        vec = np.zeros(USER_RECENCY_DIMS, dtype=np.float32)
        days_since = float(row.get("days_since_last_active", 365))
        account_age = float(row.get("account_age_days", 0))
        vec[0] = _log_scale(days_since)
        vec[1] = _log_scale(account_age)
        vec[2] = 1.0 / (1.0 + days_since)  # recency decay; peaks at 1.0 for today
        return vec

    def _sequence_embedding(self, item_ids: list) -> np.ndarray:
        """
        Mean-pool pseudo-embeddings of the last 10 clicked items.

        Production replacement: mean-pool real EventTower embeddings for the
        last 10 interacted items, loaded from the Feature Store or BigQuery.
        For sequential models, feed the sequence into a GRU encoder instead.
        """
        if not item_ids:
            return np.zeros(USER_SEQUENCE_DIMS, dtype=np.float32)
        vecs = [_hash_embedding(str(iid), USER_SEQUENCE_DIMS) for iid in item_ids[-10:]]
        return np.mean(vecs, axis=0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Event feature encoder                                                        #
# --------------------------------------------------------------------------- #

# Price bucket edges (USD). Seven edges → eight buckets: <10, 10-25, …, 200+
_PRICE_BINS = [10.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0]


class EventFeatureEncoder:
    """
    Converts a raw event/product feature dict to a 128-dim float32 numpy array.

    Input dict keys (all optional):
        category             str    Product/event category
        retail_price         float  Listed price (USD)
        cost                 float  Cost price (USD)
        margin               float  retail_price − cost
        total_unique_viewers float  All-time unique viewer count
        total_unique_buyers  float  All-time unique buyer count
        conversion_rate      float  buyers / viewers
        days_until_event     float  Days from now until the event date
        hour_of_day          int    Hour the event starts (0–23)
        is_weekend           int    1 if the event falls on Sat/Sun
        city                 str    City/venue location
        title                str    Event title or product name
    """

    def encode(self, row: dict) -> np.ndarray:
        parts = [
            _hash_embedding(row.get("category", "unknown"), EVENT_CATEGORY_DIMS),
            self._price_onehot(row.get("retail_price", 0.0)),
            self._popularity_features(row),
            self._temporal_features(row),
            _hash_embedding(row.get("city", "unknown"), EVENT_VENUE_DIMS),
            self._title_embedding(row.get("title", "")),
        ]
        vec = np.concatenate(parts).astype(np.float32)
        assert vec.shape == (128,), f"EventFeatureEncoder: expected (128,), got {vec.shape}"
        return vec

    def _price_onehot(self, price: float) -> np.ndarray:
        vec = np.zeros(EVENT_PRICE_DIMS, dtype=np.float32)
        bucket = sum(1 for b in _PRICE_BINS if float(price) >= b)
        vec[min(bucket, EVENT_PRICE_DIMS - 1)] = 1.0
        return vec

    def _popularity_features(self, row: dict) -> np.ndarray:
        vec = np.zeros(EVENT_POP_DIMS, dtype=np.float32)
        vec[0] = _log_scale(row.get("total_unique_viewers", 0))
        vec[1] = _log_scale(row.get("total_unique_buyers", 0))
        vec[2] = float(np.clip(row.get("conversion_rate", 0.0), 0.0, 1.0))
        vec[3] = _log_scale(row.get("margin", 0.0))
        return vec

    def _temporal_features(self, row: dict) -> np.ndarray:
        vec = np.zeros(EVENT_TEMPORAL_DIMS, dtype=np.float32)
        days_until = float(row.get("days_until_event", 30))
        hour = int(row.get("hour_of_day", 12))
        vec[0] = _log_scale(days_until)
        vec[1] = math.sin(2 * math.pi * hour / 24)   # cyclic encoding avoids 23→0 discontinuity
        vec[2] = math.cos(2 * math.pi * hour / 24)
        vec[3] = float(row.get("is_weekend", 0))
        vec[4] = 1.0 / (1.0 + max(days_until, 0))    # urgency: higher for imminent events
        return vec

    def _title_embedding(self, title: str) -> np.ndarray:
        """
        Bag-of-words pseudo-embedding: mean-pool per-word hash embeddings,
        then L2-normalise.

        Production replacement: call the Vertex AI Embeddings API once per
        new item and store the result in BigQuery:

            from vertexai.language_models import TextEmbeddingModel
            model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
            vec = model.get_embeddings([title])[0].values  # list[float], len=768
            # Then project to EVENT_TEXT_DIMS with a learned linear layer, or
            # truncate/PCA down to 64 dims for the event tower input.
        """
        words = str(title).lower().split() or ["unknown"]
        vecs = [_hash_embedding(w, EVENT_TEXT_DIMS) for w in words]
        vec = np.mean(vecs, axis=0).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


# --------------------------------------------------------------------------- #
# Self-test                                                                    #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    user_enc  = UserFeatureEncoder()
    event_enc = EventFeatureEncoder()

    sample_user = {
        "age": 32, "gender": "F", "country": "US",
        "top_category": "Jeans", "total_sessions": 45,
        "total_events": 312, "active_days": 28, "total_purchases": 7,
        "days_since_last_active": 3, "account_age_days": 410,
        "purchase_rate": 0.156, "recent_item_ids": ["event_1", "event_5", "event_9"],
    }
    sample_event = {
        "category": "Outerwear & Coats", "retail_price": 89.99, "cost": 42.0,
        "margin": 47.99, "total_unique_viewers": 1200, "total_unique_buyers": 84,
        "conversion_rate": 0.07, "days_until_event": 12, "hour_of_day": 19,
        "is_weekend": 1, "city": "New York", "title": "Winter Concert at Madison Square Garden",
    }

    u_vec = user_enc.encode(sample_user)
    e_vec = event_enc.encode(sample_event)

    print(f"User vector   shape: {u_vec.shape}  norm: {np.linalg.norm(u_vec):.4f}")
    print(f"Event vector  shape: {e_vec.shape}  norm: {np.linalg.norm(e_vec):.4f}")
    print(f"Dot product (raw similarity): {float(u_vec @ e_vec):.4f}")

    # Determinism check — same input must produce same output
    u_vec2 = user_enc.encode(sample_user)
    assert np.allclose(u_vec, u_vec2), "Encoder is not deterministic!"
    print("Determinism check passed.")
    print("\nAll self-tests passed.")
