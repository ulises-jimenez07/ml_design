import argparse
import json
import os
import torch
from google.cloud import storage
from google.cloud import aiplatform

# Assuming model.py is in the same directory for the EventTower class definition
from model import EventTower

def generate_and_upload_embeddings(
    bucket_name,
    prefix,
    num_events=1000,
    model_path='/tmp/model/event_tower.pth',
    embedding_dim=64,
):
    """
    Generates embeddings for the full event catalog and uploads them to GCS.

    Loads the trained EventTower from model_path when available; falls back to
    random (L2-normalised) embeddings for local testing when the file is absent.
    Output is the JSONL format required by Vertex AI Vector Search.
    """
    tower = EventTower(embedding_dim=embedding_dim)
    if os.path.exists(model_path):
        tower.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"Loaded EventTower weights from '{model_path}'.")
    else:
        print(f"Model not found at '{model_path}'. Using random weights for demonstration.")

    tower.eval()
    print(f"Generating {num_events} embeddings...")

    local_file = "/tmp/embeddings.jsonl"
    with open(local_file, "w", encoding="utf-8") as f:
        batch_size = 512
        for start in range(0, num_events, batch_size):
            end = min(start + batch_size, num_events)
            # In production, load real event feature vectors from BigQuery here.
            dummy_features = torch.randn(end - start, 128)
            with torch.no_grad():
                batch_embeddings = tower(dummy_features)
            for i, embedding in enumerate(batch_embeddings):
                record = {"id": f"event_{start + i}", "embedding": embedding.tolist()}
                f.write(json.dumps(record) + "\n")

    print(f"Wrote {num_events} embeddings to {local_file}.")

    if not bucket_name:
        print("No GCS bucket provided. Embeddings saved locally to /tmp/embeddings.jsonl.")
        return None

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"{prefix}/embeddings.jsonl")
    blob.upload_from_filename(local_file)
    upload_uri = f"gs://{bucket_name}/{prefix}/"
    print(f"Uploaded embeddings to {upload_uri}")
    return upload_uri

def deploy_vector_search(project_id, region, gcs_uri, index_name="event_index", endpoint_name="event_index_endpoint"):
    """
    Creates a Vector Search Index and deploys it to an Index Endpoint.
    """
    aiplatform.init(project=project_id, location=region)
    
    print(f"Creating Vector Search Index '{index_name}'...")
    # Creating an index can take up to an hour in a real environment.
    my_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name=index_name,
        contents_delta_uri=gcs_uri,
        dimensions=64,
        approximate_neighbors_count=10,
        distance_measure_type="DOT_PRODUCT_DISTANCE", # InfoNCE aligns with dot product/cosine
    )
    print(f"Index created: {my_index.resource_name}")

    print(f"Creating Index Endpoint '{endpoint_name}'...")
    my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
        display_name=endpoint_name,
        public_endpoint_enabled=True # For easier access in tutorial, normally VPC peered
    )
    print(f"Index Endpoint created: {my_index_endpoint.resource_name}")

    print("Deploying Index to Endpoint... (This takes a while)")
    my_index_endpoint.deploy_index(
        index=my_index,
        deployed_index_id=f"{index_name}_deployed"
    )
    print("Deployment complete.")

def add_new_event_to_index(
    event_id: str,
    event_features: list,
    index_resource_name: str,
    model_path: str = '/tmp/model/event_tower.pth',
    embedding_dim: int = 64,
):
    """
    Streams a single new-event embedding into a live Vector Search index.

    This is the item cold-start solution: when a new event is listed, call this
    function so the event is immediately recommendable without waiting for the
    next nightly index rebuild (which can take 30-60 minutes).

    Args:
        event_id:            Unique string ID for the event (must match what serving uses).
        event_features:      128-dim feature vector for the event (same schema as training).
        index_resource_name: Full resource name of the MatchingEngineIndex, e.g.
                             "projects/123/locations/us-central1/indexes/456".
        model_path:          Path to the saved EventTower state dict (.pth file).
        embedding_dim:       Embedding dimension (must match the deployed index).
    """
    tower = EventTower(embedding_dim=embedding_dim)
    tower.load_state_dict(torch.load(model_path, map_location='cpu'))
    tower.eval()

    with torch.no_grad():
        features_tensor = torch.tensor([event_features], dtype=torch.float32)
        embedding = tower(features_tensor).squeeze(0).tolist()

    aiplatform.init()
    index = aiplatform.MatchingEngineIndex(index_name=index_resource_name)
    index.upsert_datapoints(
        datapoints=[{"datapoint_id": event_id, "feature_vector": embedding}]
    )
    print(f"Upserted event '{event_id}' into index '{index_resource_name}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', type=str, default=os.environ.get('GCP_PROJECT_ID', 'YOUR_PROJECT_ID'))
    parser.add_argument('--region', type=str, default=os.environ.get('GCP_REGION', 'us-central1'))
    parser.add_argument('--gcs_bucket', type=str, default=os.environ.get('EMBEDDING_BUCKET', 'YOUR_BUCKET'))
    parser.add_argument('--gcs_prefix', type=str, default=os.environ.get('EMBEDDING_PREFIX', 'vector_search_data'))
    parser.add_argument('--num_events', type=int, default=1000)
    parser.add_argument('--model_path', type=str, default='/tmp/model/event_tower.pth',
                        help='Local path to the trained EventTower state dict.')
    # Use --mode=upsert to add a single new event to an existing live index.
    parser.add_argument('--mode', type=str, default='batch', choices=['batch', 'upsert'],
                        help='"batch": generate all embeddings and deploy index. '
                             '"upsert": add one new event to an existing live index.')
    parser.add_argument('--upsert_event_id', type=str, default=None,
                        help='Event ID to upsert (required when --mode=upsert).')
    parser.add_argument('--upsert_index_name', type=str, default=None,
                        help='Full index resource name for upsert (required when --mode=upsert).')

    args = parser.parse_args()

    if args.mode == 'upsert':
        if not args.upsert_event_id or not args.upsert_index_name:
            parser.error('--upsert_event_id and --upsert_index_name are required with --mode=upsert')
        # In production, load real event features from BigQuery or a request payload.
        # Using random features here for demonstration.
        dummy_features = torch.randn(128).tolist()
        add_new_event_to_index(
            event_id=args.upsert_event_id,
            event_features=dummy_features,
            index_resource_name=args.upsert_index_name,
            model_path=args.model_path,
        )
    else:
        upload_uri = generate_and_upload_embeddings(
            args.gcs_bucket, args.gcs_prefix, args.num_events, args.model_path
        )
        if upload_uri and args.project_id != 'YOUR_PROJECT_ID':
            deploy_vector_search(args.project_id, args.region, upload_uri)
        else:
            print("Skipping Vector Search deployment (placeholder project ID or missing GCS URI).")
