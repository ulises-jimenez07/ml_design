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

app = FastAPI(title="Event Recommendation API")

# --- Environment Variables ---
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "YOUR_PROJECT_ID")
REGION = os.environ.get("GCP_REGION", "us-central1")
FEATURE_STORE_NAME = os.environ.get("FEATURE_STORE_NAME", "event_feature_store")
FEATURE_VIEW_NAME = os.environ.get("FEATURE_VIEW_NAME", "user_features_view")
INDEX_ENDPOINT_ID = os.environ.get("INDEX_ENDPOINT_ID", "YOUR_INDEX_ENDPOINT_ID")
DEPLOYED_INDEX_ID = os.environ.get("DEPLOYED_INDEX_ID", "event_index_deployed")

# --- Globals for Cold Start / Caching ---
model = None
feature_store_client = None
index_endpoint = None

class RecommendationRequest(BaseModel):
    user_id: str
    num_recommendations: int = 10

def load_model():
    """Loads the PyTorch User Tower model."""
    global model
    if model is None:
        print("Loading User Tower model (Cold Start)...")
        # In a real app, you would download the model.pth from GCS here.
        # For this example, we initialize a fresh model to simulate loading.
        model = UserTower(embedding_dim=64)
        model.eval()
        print("Model loaded successfully.")

def init_gcp_clients():
    """Initializes Vertex AI clients."""
    global feature_store_client, index_endpoint
    if feature_store_client is None:
        print("Initializing GCP Clients (Cold Start)...")
        aiplatform.init(project=PROJECT_ID, location=REGION)
        
        # Next-Gen Feature Store Client
        client_options = {"api_endpoint": f"{REGION}-aiplatform.googleapis.com"}
        feature_store_client = FeatureOnlineStoreServiceClient(client_options=client_options)
        
        # Vector Search Client
        if INDEX_ENDPOINT_ID != "YOUR_INDEX_ENDPOINT_ID":
            index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=INDEX_ENDPOINT_ID)
        print("GCP Clients initialized.")

@app.on_event("startup")
async def startup_event():
    """Handle cold start by pre-loading resources on startup."""
    load_model()
    # Note: We don't initialize GCP clients strictly on startup to allow local 
    # testing without credentials if endpoints aren't hit.
    # init_gcp_clients()

@retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3))
def fetch_user_features(user_id: str):
    """Fetches real-time user features from Vertex AI Feature Store."""
    if PROJECT_ID == "YOUR_PROJECT_ID":
        # Simulate feature retrieval if project is placeholder
        return torch.randn(1, 128)
        
    init_gcp_clients()
    
    feature_view_path = feature_store_client.feature_view_path(
        PROJECT_ID, REGION, FEATURE_STORE_NAME, FEATURE_VIEW_NAME
    )
    
    request = feature_online_store_service.FetchFeatureValuesRequest(
        feature_view=feature_view_path,
        data_key=feature_online_store_service.FeatureViewDataKey(key=user_id)
    )
    
    response = feature_store_client.fetch_feature_values(request=request)
    
    # In a real app, parse response.key_values to a tensor
    # Simulated for now:
    return torch.randn(1, 128)

@retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3))
def query_vector_search(embedding: list, num_neighbors: int):
    """Queries Vertex AI Vector Search."""
    if index_endpoint is None:
        # Simulate response
        return [{"id": f"event_{i}", "distance": 0.9 - (i*0.01)} for i in range(num_neighbors)]

    response = index_endpoint.find_neighbors(
        deployed_index_id=DEPLOYED_INDEX_ID,
        queries=[embedding],
        num_neighbors=num_neighbors
    )
    
    recommendations = []
    for neighbor in response[0]:
        recommendations.append({
            "id": neighbor.id,
            "distance": neighbor.distance
        })
    return recommendations

@app.post("/recommend")
async def get_recommendations(request: RecommendationRequest):
    try:
        # 1. Fetch Features
        start_time = time.time()
        features = fetch_user_features(request.user_id)
        
        # 2. Inference (Get User Embedding)
        load_model()
        with torch.no_grad():
            user_embedding = model(features).squeeze(0).tolist()
            
        # 3. Vector Search
        recommendations = query_vector_search(user_embedding, request.num_recommendations)
        
        latency = time.time() - start_time
        
        return {
            "user_id": request.user_id,
            "recommendations": recommendations,
            "latency_seconds": round(latency, 4)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
