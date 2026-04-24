import argparse
import json
import os
import torch
from google.cloud import storage
from google.cloud import aiplatform

# Assuming model.py is in the same directory for the EventTower class definition
from model import EventTower

def generate_and_upload_embeddings(bucket_name, prefix, num_events=1000, embedding_dim=64):
    """
    Simulates batch embedding generation for the event catalog using the trained Event Tower.
    Saves embeddings in the JSONL format required by Vertex AI Vector Search.
    """
    print(f"Generating {num_events} embeddings...")
    
    # In a real scenario, we would load the trained EventTower state_dict
    # and run a forward pass on actual event features.
    # Here, we generate random embeddings for demonstration.
    embeddings = torch.randn(num_events, embedding_dim)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    local_file = "/tmp/embeddings.jsonl"
    with open(local_file, "w") as f:
        for i in range(num_events):
            record = {
                "id": f"event_{i}",
                "embedding": embeddings[i].tolist()
            }
            f.write(json.dumps(record) + "\n")
            
    if not bucket_name:
        print("No GCS bucket provided, embeddings saved to /tmp/embeddings.jsonl")
        return None

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"{prefix}/embeddings.jsonl")
    blob.upload_from_filename(local_file)
    gcs_uri = f"gs://{bucket_name}/{prefix}/"
    print(f"Uploaded embeddings to {gcs_uri}")
    return gcs_uri

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', type=str, default=os.environ.get('GCP_PROJECT_ID', 'YOUR_PROJECT_ID'))
    parser.add_argument('--region', type=str, default=os.environ.get('GCP_REGION', 'us-central1'))
    parser.add_argument('--gcs_bucket', type=str, default=os.environ.get('EMBEDDING_BUCKET', 'YOUR_BUCKET'))
    parser.add_argument('--gcs_prefix', type=str, default=os.environ.get('EMBEDDING_PREFIX', 'vector_search_data'))
    parser.add_argument('--num_events', type=int, default=1000)
    
    args = parser.parse_args()
    
    gcs_uri = generate_and_upload_embeddings(args.gcs_bucket, args.gcs_prefix, args.num_events)
    
    if gcs_uri and args.project_id != 'YOUR_PROJECT_ID':
        deploy_vector_search(args.project_id, args.region, gcs_uri)
    else:
        print("Skipping Vector Search deployment because Project ID is a placeholder or GCS URI is missing.")
