# Machine Learning System Design: Real-Time Event Recommendation

This document breaks down the design decisions, trade-offs, and architecture of the Real-Time Event Recommendation System, structured similarly to a Machine Learning System Design interview.

---

## 1. Problem Formulation

**Objective:** Build a highly scalable, real-time recommendation engine that suggests relevant events to users based on their historical preferences and live interactions (clicks).

*   **Inputs:** 
    *   User features (Historical interactions, demographics, location).
    *   Event features (Category, price, popularity, metadata).
    *   Real-time context (Recent clicks, time of day).
*   **Outputs:** Top-K recommended events for a given user.
*   **Business Metrics:** Click-Through Rate (CTR), Conversion Rate (Ticket sales/RSVPs), User Engagement (Time spent browsing events).
*   **ML Metrics:** Recall@K, Normalized Discounted Cumulative Gain (NDCG), Mean Reciprocal Rank (MRR).

---

## 2. Data & Feature Engineering

### Feature Representation
*   **User Features:** Encoded as a dense vector. Incorporates static features (e.g., user age, base location) and dynamic features (e.g., recent event categories interacted with).
*   **Event Features:** Encoded as a dense vector. Incorporates categorical data (embeddings for event type/category), numerical data (price, duration), and dynamic data (rolling click counts).

### Real-Time Streaming Pipeline
To capture the dynamic popularity of events, we need real-time features.
*   **Design Decision:** We use **Apache Beam (Dataflow)** to listen to a Pub/Sub topic for live click streams.
*   **Aggregation:** We apply a 5-minute sliding window to aggregate clicks per event.
*   **Storage:** These features are continuously upserted into **Vertex AI Feature Store (Next Gen)**, which is backed by BigQuery.
*   **Why?** This allows the serving layer to instantly know if an event is currently "trending" without waiting for a nightly batch job.

---

## 3. Model Architecture: Two-Tower Neural Network

For the recommendation engine, we chose a **Two-Tower (Dual Encoder) Architecture**.

### Why Two-Tower?
*   **Scalability:** In a system with millions of events, scoring every user-event pair at runtime with a complex cross-feature model (like DeepFM) is computationally infeasible.
*   **Decoupling:** Two-Tower allows us to pre-compute and index the Event embeddings offline. At inference time, we only need to pass the User features through the User Tower and perform a fast Nearest Neighbor search.
*   **Alternative Considered:** Matrix Factorization. While simple, it struggles to incorporate rich, heterogeneous side-features (like real-time popularity or user metadata). Two-Tower easily absorbs diverse feature types via multi-layer perceptrons (MLPs).

### Loss Function: Contrastive Loss (InfoNCE)
*   Instead of framing this purely as binary classification (clicked vs. not clicked), we treat it as a representation learning problem.
*   **InfoNCE (Normalized Temperature-scaled Cross Entropy):** We want the dot product of a user embedding and their interacted event embedding to be high, while the dot product with *all other* events in the batch (in-batch negatives) should be low.
*   **Why?** This naturally pushes similar users and events together in the embedding space and is highly efficient because it utilizes other samples in the training batch as negative examples without requiring explicit negative sampling algorithms.

---

## 4. Training & Offline Evaluation

### Training Strategy
*   **Vertex AI Custom Training:** The model is trained using PyTorch on GPUs.
*   **Batching Strategy:** Because we rely on in-batch negatives for Contrastive Loss, a larger batch size (e.g., 2048 or 4096) is crucial. It ensures a diverse set of negative examples for every positive pair.

### Offline Evaluation
Before deploying, the model is evaluated on a holdout validation set:
*   **Recall@K:** Out of all the items the user actually clicked in the validation set, what percentage appeared in the top K recommendations?
*   **NDCG:** Evaluates the ranking quality—getting a relevant event at position 1 is better than getting it at position 10.

---

## 5. Serving & Online Infrastructure

The serving phase is optimized for low latency (< 100ms). It represents the **Candidate Generation** phase of a classic recommendation funnel.

### Inference Flow
1.  **Feature Retrieval:** The FastAPI service receives a `user_id` and queries the **Vertex AI Feature Store** for the latest user features.
2.  **User Inference:** The user features are passed through the deployed `UserTower` to generate a live, 64-dimensional User Embedding.
3.  **Vector Search (ANN):** The User Embedding is sent to the **Vertex AI Vector Search** index. The index uses Approximate Nearest Neighbor (Tree-AH) algorithms to search through millions of pre-computed Event embeddings in milliseconds, returning the top K nearest neighbors using Dot Product distance.

### Handling Cold Starts
*   **System Cold Start (API Startup):** The serving API lazily loads the PyTorch model and Vertex AI clients into memory upon startup or on the first request to prevent slow individual request processing.
*   **User Cold Start (New Users):** For users with no history, the User Tower can fall back to using default demographic averages or pure real-time context (e.g., location, time of day) to generate a generalized embedding.
*   **Item Cold Start (New Events):** When a new event is added, its embedding is generated using the `EventTower` and instantly upserted to the Vector Search index using its Streaming Update capability, allowing it to be recommended immediately.

---

## 6. Trade-offs & Future Improvements

### 1. Adding a Ranking Stage
*   **Current State:** The Two-Tower model serves as a highly scalable Candidate Generator. However, the dot product distance isn't perfect for capturing complex, non-linear interactions between user and event features.
*   **Improvement:** Introduce a heavier **Ranking Stage** (e.g., DCN - Deep Cross Network or XGBoost). The Vector Search would return the top 500 candidates, and the Ranker would score them individually to find the absolute top 10, trading off slight latency for a massive boost in accuracy.

### 2. Handling Data Skew (Popularity Bias)
*   **Current State:** In-batch negative sampling can accidentally penalize highly popular events because they appear frequently as "negatives" for other users.
*   **Improvement:** Implement **LogQ Correction**. We subtract the log of the item's probability of appearing in a batch from the logits before calculating the loss, preventing popular items from being overly penalized.

### 3. Asynchronous Feature Fetching
*   **Current State:** The FastAPI app fetches features synchronously.
*   **Improvement:** Use Python's `asyncio` to fetch user features and contextual features in parallel, further reducing the overall request latency.
