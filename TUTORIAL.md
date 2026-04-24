# Real-Time Event Recommendation System on GCP

This tutorial guides you through deploying the Two-Tower Event Recommendation System on Google Cloud Platform. 
The system relies on Dataflow for real-time feature streaming, Vertex AI Custom Training for the PyTorch model, Vertex AI Vector Search for retrieval, and Cloud Run for the FastAPI serving layer.

## Prerequisites

1.  A GCP Project with billing enabled.
2.  `gcloud` CLI installed and authenticated.
3.  Python 3.9+ installed locally.

## Step 1: Set Up Environment Variables

We use environment variables across all scripts to ensure portability. Replace the placeholders with your actual GCP resource names.

```bash
export GCP_PROJECT_ID="your-project-id"
export GCP_REGION="us-central1"
export MODEL_BUCKET="your-model-bucket"
export EMBEDDING_BUCKET="your-embedding-bucket"
export PIPELINE_ROOT="gs://$MODEL_BUCKET/pipeline_root"
export PUBSUB_TOPIC="projects/$GCP_PROJECT_ID/topics/event-topic"
export PUBSUB_SUBSCRIPTION="projects/$GCP_PROJECT_ID/subscriptions/event-sub"
export BQ_FEATURE_TABLE="$GCP_PROJECT_ID:feature_store.event_features_rt"
```

Set up your default credentials:
```bash
gcloud config set project $GCP_PROJECT_ID
gcloud auth application-default login
```

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 3: Start the Dataflow Pipeline (Feature Engineering)

This pipeline listens to Pub/Sub events and pushes aggregated features to BigQuery (which backs the Vertex AI Next-Gen Feature Store).

1.  Create the Pub/Sub topic and subscription:
    ```bash
    gcloud pubsub topics create event-topic
    gcloud pubsub subscriptions create event-sub --topic=event-topic
    ```
2.  Run the pipeline:
    ```bash
    python dataflow_pipeline.py \
      --input_subscription=$PUBSUB_SUBSCRIPTION \
      --output_table=$BQ_FEATURE_TABLE
    ```
    *(Note: To run on Dataflow instead of DirectRunner, append `--runner=DataflowRunner --project=$GCP_PROJECT_ID --region=$GCP_REGION --temp_location=gs://$MODEL_BUCKET/temp`)*

## Step 4: Run the Vertex AI Pipeline (Training & Indexing)

The ML pipeline orchestrates model training and Vector Search index deployment.

```bash
python pipeline.py
```
This script will compile the KFP pipeline to `recommendation_pipeline.json` and submit it to Vertex AI Pipelines. You can monitor the progress in the GCP Console under **Vertex AI > Pipelines**.

*Note: Vector Search Index deployment can take up to an hour.*

## Step 5: Deploy the Serving Layer to Cloud Run

Once the Vector Search endpoint is up, capture its ID to configure the serving service.

```bash
export INDEX_ENDPOINT_ID="your-deployed-index-endpoint-id"
export DEPLOYED_INDEX_ID="event_index_deployed"
```

Build and deploy the FastAPI Docker container:

```bash
gcloud builds submit --tag gcr.io/$GCP_PROJECT_ID/serving-api

gcloud run deploy serving-api \
  --image gcr.io/$GCP_PROJECT_ID/serving-api \
  --platform managed \
  --region $GCP_REGION \
  --allow-unauthenticated \
  --set-env-vars GCP_PROJECT_ID=$GCP_PROJECT_ID,GCP_REGION=$GCP_REGION,INDEX_ENDPOINT_ID=$INDEX_ENDPOINT_ID,DEPLOYED_INDEX_ID=$DEPLOYED_INDEX_ID
```

## Step 6: Test the System

Once Cloud Run provides a URL, test the `/recommend` endpoint:

```bash
curl -X POST https://<CLOUD_RUN_URL>/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_123", "num_recommendations": 10}'
```

You should receive a JSON response containing top 10 recommended events and the serving latency.
