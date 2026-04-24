# Real-Time Event Recommendation Engine on GCP

This repository contains a reference implementation for a production-ready **Real-Time Event Recommendation Engine** on Google Cloud Platform (GCP). It demonstrates how to build, orchestrate, and deploy a scalable recommendation system using a **Two-Tower Neural Network** architecture.

## Overview

The system is designed to handle high-concurrency retrieval and real-time feature updates. It leverages the following core technologies:

*   **Google Cloud Platform (GCP)**: Vertex AI, Cloud Run, Dataflow, Cloud Storage, Pub/Sub, BigQuery.
*   **Machine Learning**: PyTorch (Two-Tower Architecture, InfoNCE Contrastive Loss).
*   **MLOps**: Kubeflow Pipelines (KFP) for orchestration.
*   **Serving**: FastAPI for high-performance online inference.

## Key Components

1.  **Feature Engineering (`dataflow_pipeline.py`)**: An Apache Beam streaming pipeline that listens to Pub/Sub for live "click" events, aggregates them over a sliding window, and ingests them into Vertex AI Feature Store.
2.  **Model Development (`model.py`)**: A PyTorch implementation of the Two-Tower architecture (User Tower & Event Tower) optimized with Contrastive Loss.
3.  **Indexing (`index_deploy.py`)**: Batch generation of event embeddings and deployment to a Vertex AI Vector Search Index for ultra-fast Approximate Nearest Neighbor (ANN) retrieval.
4.  **Serving (`serving.py`, `Dockerfile`)**: A low-latency FastAPI service deployable to Cloud Run that fetches real-time features, runs the User Tower inference, and queries the Vector Search endpoint.
5.  **Orchestration (`pipeline.py`)**: A Vertex AI Pipeline (KFP) script that automates model training and index deployment.

## Documentation

For a deeper dive into the system, please refer to the following documents:

*   📖 **[ML System Design Document](ML_SYSTEM_DESIGN.md)**: Explains the problem formulation, architecture choices (why Two-Tower?), loss functions, online/offline metrics, and tradeoffs. Read this for the "why" behind the code.
*   🚀 **[Deployment Tutorial](TUTORIAL.md)**: A step-by-step guide to setting up your environment variables, running the Dataflow and ML pipelines, and deploying the serving API on GCP. Read this for the "how-to".

## Prerequisites

- GCP Account with Billing Enabled
- `gcloud` CLI installed
- Python 3.9+
- See `requirements.txt` for Python dependencies.
