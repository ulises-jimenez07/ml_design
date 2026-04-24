import os
from kfp import dsl
from kfp import compiler
from google.cloud import aiplatform

# --- Environment Variables ---
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "YOUR_PROJECT_ID")
REGION = os.environ.get("GCP_REGION", "us-central1")
PIPELINE_ROOT = os.environ.get("PIPELINE_ROOT", "gs://YOUR_BUCKET/pipeline_root")
MODEL_BUCKET = os.environ.get("MODEL_BUCKET", "YOUR_BUCKET")
EMBEDDING_BUCKET = os.environ.get("EMBEDDING_BUCKET", "YOUR_BUCKET")
DOCKER_IMAGE = os.environ.get("TRAINING_IMAGE", "us-docker.pkg.dev/vertex-ai/training/pytorch-xla.1-13:latest")

@dsl.component(base_image="python:3.9", packages_to_install=["google-cloud-aiplatform", "torch", "google-cloud-storage"])
def train_model_op(
    project_id: str,
    region: str,
    model_bucket: str,
    prefix: str,
    docker_image: str
) -> str:
    """
    Submits a Vertex AI Custom Training Job.
    In a real scenario, this would use google-cloud-aiplatform SDK to submit
    the model.py script. We simulate it here by logging.
    """
    import logging
    logging.info(f"Submitting training job to Vertex AI in {region} for project {project_id}...")
    logging.info(f"Using image {docker_image} and saving to gs://{model_bucket}/{prefix}")
    
    # Simulate training time
    # In practice: job.run(sync=True)
    
    gcs_model_uri = f"gs://{model_bucket}/{prefix}"
    logging.info("Training complete.")
    return gcs_model_uri

@dsl.component(base_image="python:3.9", packages_to_install=["google-cloud-aiplatform", "torch", "google-cloud-storage"])
def deploy_index_op(
    project_id: str,
    region: str,
    embedding_bucket: str,
    prefix: str,
    num_events: int
) -> str:
    """
    Calls the logic from index_deploy.py to generate embeddings and deploy the index.
    """
    import logging
    logging.info("Generating embeddings and deploying Vector Search Index...")
    
    # In a real scenario, we would import from index_deploy or copy the logic here.
    logging.info(f"Targeting bucket gs://{embedding_bucket}/{prefix} for {num_events} events.")
    logging.info(f"Project: {project_id}, Region: {region}")
    
    return "event_index_endpoint_id"

@dsl.pipeline(
    name="two-tower-recommendation-pipeline",
    description="Trains the Two-Tower model and deploys the Event Tower embeddings to Vector Search",
    pipeline_root=PIPELINE_ROOT,
)
def recommendation_pipeline(
    project_id: str = PROJECT_ID,
    region: str = REGION,
    model_bucket: str = MODEL_BUCKET,
    model_prefix: str = "recommendation_model",
    embedding_bucket: str = EMBEDDING_BUCKET,
    embedding_prefix: str = "vector_search_data",
    num_events: int = 1000,
    docker_image: str = DOCKER_IMAGE
):
    # Step 1: Train Model
    train_task = train_model_op(
        project_id=project_id,
        region=region,
        model_bucket=model_bucket,
        prefix=model_prefix,
        docker_image=docker_image
    )
    
    # Step 2: Generate Embeddings & Deploy Index (Dependent on Training)
    deploy_task = deploy_index_op(
        project_id=project_id,
        region=region,
        embedding_bucket=embedding_bucket,
        prefix=embedding_prefix,
        num_events=num_events
    )
    deploy_task.after(train_task)

def compile_and_run():
    # Compile the pipeline
    compiler.Compiler().compile(
        pipeline_func=recommendation_pipeline,
        package_path="recommendation_pipeline.json"
    )
    
    print("Pipeline compiled to recommendation_pipeline.json")
    
    if PROJECT_ID != "YOUR_PROJECT_ID":
        print("Submitting pipeline to Vertex AI...")
        aiplatform.init(project=PROJECT_ID, location=REGION)
        
        job = aiplatform.PipelineJob(
            display_name="two-tower-pipeline",
            template_path="recommendation_pipeline.json",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=True
        )
        job.submit()
        print(f"Pipeline job submitted. View it at: {job.dashboard_uri}")
    else:
        print("Skipping submission because PROJECT_ID is a placeholder.")

if __name__ == "__main__":
    compile_and_run()
