"""FastAPI serving layer for the Two-Tower event recommendation system."""

import asyncio
import json
import logging
import os
import time
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import aiplatform
from google.cloud.aiplatform_v1beta1 import FeatureOnlineStoreServiceClient
from google.cloud.aiplatform_v1beta1.types import feature_online_store_service
from tenacity import retry, wait_exponential, stop_after_attempt

from model import UserTower

logger = logging.getLogger(__name__)
app = FastAPI(title="Event Recommendation API")

# --- Environment Variables ---
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "YOUR_PROJECT_ID")
REGION = os.environ.get("GCP_REGION", "us-central1")
FEATURE_STORE_NAME = os.environ.get("FEATURE_STORE_NAME", "event_feature_store")
FEATURE_VIEW_NAME = os.environ.get("FEATURE_VIEW_NAME", "user_features_view")
INDEX_ENDPOINT_ID = os.environ.get("INDEX_ENDPOINT_ID", "YOUR_INDEX_ENDPOINT_ID")
DEPLOYED_INDEX_ID = os.environ.get("DEPLOYED_INDEX_ID", "event_index_deployed")
# A/B testing: set per Cloud Run revision so outcomes can be joined in BigQuery.
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1")
EXPERIMENT_ID = os.environ.get("EXPERIMENT_ID", "none")
# Optional Pub/Sub topic for recommendation logs. Leave unset to disable.
LOG_TOPIC = os.environ.get("RECOMMENDATION_LOG_TOPIC", "")

# --- Module-level singletons (populated on first use to allow local testing) ---
_model = None
_feature_store_client = None
_index_endpoint = None
_pubsub_publisher = None


class RecommendationRequest(BaseModel):
    """Request body for the /recommend endpoint."""

    user_id: str
    num_recommendations: int = 10


# ---------------------------------------------------------------------------
# Initialisation helpers
# ---------------------------------------------------------------------------

def _load_model():
    global _model
    if _model is None:
        logger.info("Loading User Tower model (cold start)...")
        # In production: download user_tower.pth from GCS before loading.
        _model = UserTower(embedding_dim=64)
        _model.eval()
        logger.info("User Tower model loaded.")


def _init_gcp_clients():
    global _feature_store_client, _index_endpoint
    if _feature_store_client is None:
        logger.info("Initialising GCP clients (cold start)...")
        aiplatform.init(project=PROJECT_ID, location=REGION)

        client_options = {"api_endpoint": f"{REGION}-aiplatform.googleapis.com"}
        _feature_store_client = FeatureOnlineStoreServiceClient(client_options=client_options)

        if INDEX_ENDPOINT_ID != "YOUR_INDEX_ENDPOINT_ID":
            _index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
                index_endpoint_name=INDEX_ENDPOINT_ID
            )
        logger.info("GCP clients initialised.")


def _get_pubsub_publisher():
    """Lazily initialise the Pub/Sub publisher. Returns None if topic is unset."""
    global _pubsub_publisher
    if not LOG_TOPIC:
        return None
    if _pubsub_publisher is None:
        from google.cloud import pubsub_v1  # optional dependency
        _pubsub_publisher = pubsub_v1.PublisherClient()
    return _pubsub_publisher


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """Pre-load the PyTorch model at startup to avoid per-request cold start."""
    _load_model()
    # GCP clients are initialised lazily on first request so that the container
    # can start and pass health checks even without valid credentials locally.


# ---------------------------------------------------------------------------
# Feature retrieval
# ---------------------------------------------------------------------------

@retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3))
def fetch_user_features(user_id: str) -> torch.Tensor:
    """Fetches real-time user features from Vertex AI Feature Store."""
    if PROJECT_ID == "YOUR_PROJECT_ID":
        return torch.randn(1, 128)

    _init_gcp_clients()

    feature_view_path = _feature_store_client.feature_view_path(
        PROJECT_ID, REGION, FEATURE_STORE_NAME, FEATURE_VIEW_NAME
    )
    request = feature_online_store_service.FetchFeatureValuesRequest(
        feature_view=feature_view_path,
        data_key=feature_online_store_service.FeatureViewDataKey(key=user_id),
    )
    # response.key_values contains the feature values; parse to a tensor in production.
    _feature_store_client.fetch_feature_values(request=request)
    return torch.randn(1, 128)  # placeholder until response parsing is implemented


# ---------------------------------------------------------------------------
# Vector Search
# ---------------------------------------------------------------------------

@retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3))
def query_vector_search(embedding: list, num_neighbors: int) -> list:
    """Queries Vertex AI Vector Search (ScaNN) for the nearest event embeddings."""
    if _index_endpoint is None:
        return [{"id": f"event_{i}", "distance": round(0.9 - i * 0.01, 3)}
                for i in range(num_neighbors)]

    response = _index_endpoint.find_neighbors(
        deployed_index_id=DEPLOYED_INDEX_ID,
        queries=[embedding],
        num_neighbors=num_neighbors,
    )
    return [{"id": neighbor.id, "distance": neighbor.distance} for neighbor in response[0]]


# ---------------------------------------------------------------------------
# Recommendation logging (fire-and-forget, for A/B analysis)
# ---------------------------------------------------------------------------

async def _log_recommendation(user_id: str, recommendations: list, latency: float):
    """
    Publishes a recommendation log entry to Pub/Sub for A/B experiment analysis.

    This runs as a background asyncio task — it does not block the response.
    A Dataflow job can consume these logs and join them with click events in
    BigQuery to compute per-experiment CTR and conversion rates.
    """
    publisher = _get_pubsub_publisher()
    if publisher is None:
        return

    payload = json.dumps({
        "user_id": user_id,
        "model_version": MODEL_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "recommended_ids": [r["id"] for r in recommendations],
        "latency_seconds": latency,
        "timestamp": time.time(),
    }).encode()

    try:
        publisher.publish(LOG_TOPIC, payload)
    except Exception:
        logger.warning("Failed to publish recommendation log.", exc_info=True)


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------

@app.post("/recommend")
async def get_recommendations(request: RecommendationRequest):
    """Return top-K event recommendations for a given user."""
    try:
        start_time = time.time()

        # 1. Fetch real-time user features from Feature Store
        features = fetch_user_features(request.user_id)

        # 2. Run User Tower to get the user embedding
        _load_model()
        with torch.no_grad():
            user_embedding = _model(features).squeeze(0).tolist()

        # 3. Approximate Nearest Neighbour search over pre-computed event embeddings
        recommendations = query_vector_search(user_embedding, request.num_recommendations)

        latency = round(time.time() - start_time, 4)

        # 4. Log asynchronously — does not add to response latency
        asyncio.create_task(_log_recommendation(request.user_id, recommendations, latency))

        return {
            "user_id": request.user_id,
            "recommendations": recommendations,
            "latency_seconds": latency,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
