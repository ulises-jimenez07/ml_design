# Real-Time Event Recommendation System: A Complete ML System Design Tutorial

> **Structure:** This tutorial follows the framework from *Machine Learning System Design Interview* by Ali Aminian and Alex Xu. Each decision is explained with alternatives considered, trade-offs accepted, and production concerns addressed. Code references are grounded in the actual source files in this repository.

---

## Table of Contents

1. [Problem Clarification & Requirements](#1-problem-clarification--requirements)
2. [Scale Estimation & SLAs](#2-scale-estimation--slas)
3. [System Architecture Overview](#3-system-architecture-overview)
4. [Data Strategy: Exploring Public GCP Data](#4-data-strategy-exploring-public-gcp-data)
5. [Feature Engineering](#5-feature-engineering)
6. [Model Architecture: Two-Tower Deep Dive](#6-model-architecture-two-tower-deep-dive)
7. [Training Pipeline](#7-training-pipeline)
8. [Offline Evaluation](#8-offline-evaluation)
9. [Infrastructure Setup (GCP)](#9-infrastructure-setup-gcp)
10. [Running the Dataflow Pipeline](#10-running-the-dataflow-pipeline)
11. [Running the Vertex AI Training Pipeline](#11-running-the-vertex-ai-training-pipeline)
12. [Deploying the Serving Layer](#12-deploying-the-serving-layer)
13. [Online Evaluation & A/B Testing](#13-online-evaluation--ab-testing)
14. [Monitoring & Observability](#14-monitoring--observability)
15. [Trade-offs & Alternatives](#15-trade-offs--alternatives)

---

## 1. Problem Clarification & Requirements

In an ML System Design interview, you never jump straight to the model. You first clarify the problem by asking questions. Here is how a structured clarification would unfold.

### Clarifying Questions

**Q: What kind of events are we recommending?**
A: Live events — concerts, sports games, local meetups, conferences. Each event has a fixed capacity and a date, unlike a product that can be purchased anytime.

**Q: What is the primary objective of the recommendation?**
A: Maximize the number of ticket purchases (conversions). Click-through-rate (CTR) is a leading indicator, but the end goal is a sold ticket.

**Q: How many users and events are in the system?**
A: ~10 million monthly active users, ~1 million events active at any given time.

**Q: What is the acceptable latency for a recommendation request?**
A: The recommendation must be served within **150ms** end-to-end (P99). The UI shows recommendations below the fold, giving a tight budget.

**Q: Do we have explicit feedback (star ratings) or implicit feedback (clicks)?**
A: Only implicit feedback — clicks, purchases, and time spent viewing an event listing. No explicit ratings.

**Q: How fresh do real-time features need to be?**
A: Features should reflect activity within the last 5 minutes. A concert going viral on social media should surface quickly.

**Q: Is there a cold start problem?**
A: Yes, both user cold start (new user signup) and item cold start (new event listed hours before show).

### Functional Requirements

| Requirement | Detail |
|---|---|
| Recommend K events per user | K = 10 (default), up to 50 |
| Support real-time feature freshness | < 5-minute lag for trending signals |
| Handle new events | New event must be recommendable within 10 minutes of creation |
| Handle new users | Degrade gracefully for users with no history |

### Non-Functional Requirements

| Requirement | Target |
|---|---|
| Recommendation latency (P99) | < 150ms |
| System availability | 99.9% uptime |
| Training freshness | Daily model retraining |
| Feature freshness | < 5 minutes |
| Throughput | 50,000 recommendation requests/second at peak |

---

## 2. Scale Estimation & SLAs

Understanding scale shapes every subsequent decision. Under-estimating leads to a system that falls over in production.

```
Users:         10M monthly active users
Events:        1M active events (index size for Vector Search)
Peak RPS:      50,000 requests/second
Avg latency:   < 150ms (P99)
Embedding dim: 64 (chosen for balance of quality vs. ANN search speed)
Event churn:   ~10,000 new events/day (streaming upserts to index)

Data storage:
  Event embeddings: 1M events × 64 dims × 4 bytes = 256 MB (fits in RAM on index nodes)
  Training data:    ~100M user-event interaction rows/day → ~50 GB/day in BigQuery
  Feature store:    ~10M users × 50 features × 8 bytes = 4 GB (user features)
```

**Why these numbers matter for architecture:**

- **1M events × 64 dims** means the Vertex AI Vector Search index is small enough to use the Tree-AH (ScaNN-based) algorithm and return ANN results in < 10ms.
- **50,000 RPS** means the Cloud Run serving layer needs horizontal auto-scaling. Each replica handles ~500 RPS, so we need ~100 replicas at peak.
- **5-minute feature freshness** rules out a nightly batch pipeline and mandates streaming (Dataflow + Pub/Sub).

---

## 3. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ONLINE PATH (< 150ms)                        │
│                                                                     │
│  User Request                                                       │
│      │                                                              │
│      ▼                                                              │
│  Cloud Run (FastAPI)                                                │
│      │                                                              │
│      ├──► Vertex AI Feature Store ──► User Embedding (UserTower)   │
│      │    (< 10ms lookup)              (PyTorch inference, < 5ms)  │
│      │                                                              │
│      └──► Vertex AI Vector Search ──► Top-K Event IDs              │
│           (Tree-AH ANN, < 20ms)        (returned to client)        │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                   REAL-TIME FEATURE PIPELINE                        │
│                                                                     │
│  User Clicks                                                        │
│      │                                                              │
│      ▼                                                              │
│  Cloud Pub/Sub ──► Dataflow (Apache Beam)                          │
│                    │  5-min sliding windows                         │
│                    │  click_count_5m per event_id                   │
│                    ▼                                                │
│              BigQuery Feature Table ──► Vertex AI Feature Store     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                   OFFLINE TRAINING PIPELINE                         │
│                                                                     │
│  BigQuery (raw events + static features)                            │
│      │                                                              │
│      ▼                                                              │
│  Vertex AI Pipelines (KFP)                                         │
│      │                                                              │
│      ├──► Custom Training Job (PyTorch, GPU)                       │
│      │    model.py: TwoTowerModel, InfoNCE loss                    │
│      │                                                              │
│      └──► Index Deploy Job                                         │
│           index_deploy.py: EventTower → embeddings → Vector Search  │
└─────────────────────────────────────────────────────────────────────┘
```

**Design Philosophy:** This is a classic two-stage retrieval architecture used at Pinterest, YouTube, and Spotify:
- **Stage 1 (Candidate Generation):** Two-Tower + ANN returns top 500 candidates cheaply.
- **Stage 2 (Ranking):** A heavier model scores the 500 candidates. *(See Section 15 for the ranking stage design.)*

---

## 4. Data Strategy: Exploring Public GCP Data

Before writing a single line of model code, you need to understand your data. This section uses the **`bigquery-public-data.thelook_ecommerce`** dataset — a realistic synthetic e-commerce dataset available to all GCP users at no cost — to demonstrate data exploration, label generation, and feature extraction patterns that map directly to our event recommendation problem.

> **Mapping to event recommendation:** In theLook, "products" map to "events" and "users" map to our users. The `events` table contains interaction signals (views, clicks, purchases) that serve as implicit feedback.

### 4.1 Explore the Dataset

Run these queries in BigQuery to understand the data before designing features.

**Check available tables:**
```sql
SELECT table_name, row_count
FROM `bigquery-public-data.thelook_ecommerce`.INFORMATION_SCHEMA.TABLES
ORDER BY row_count DESC;
```

Expected output:
```
table_name          | row_count
--------------------|----------
events              | 2,400,000+
orders              | 125,000+
order_items         | 181,000+
users               | 100,000+
products            | 29,120
```

**Understand interaction types (analogous to event clicks in our system):**
```sql
SELECT
    event_type,
    COUNT(*) AS event_count,
    COUNT(DISTINCT user_id) AS unique_users,
    COUNT(DISTINCT session_id) AS unique_sessions
FROM `bigquery-public-data.thelook_ecommerce.events`
GROUP BY event_type
ORDER BY event_count DESC;
```

This tells you the distribution of implicit feedback signals. In a real events platform, you'd see analogous signals: `view_event_page`, `click_buy_tickets`, `purchase_complete`.

**Analyze temporal patterns (critical for windowed features):**
```sql
SELECT
    DATE(created_at) AS date,
    event_type,
    COUNT(*) AS events_per_day
FROM `bigquery-public-data.thelook_ecommerce.events`
WHERE created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
GROUP BY date, event_type
ORDER BY date DESC, events_per_day DESC;
```

**Identify the cold start problem magnitude:**
```sql
-- What fraction of users have fewer than 3 interactions? (Cold users)
WITH user_interaction_counts AS (
    SELECT user_id, COUNT(*) AS num_interactions
    FROM `bigquery-public-data.thelook_ecommerce.events`
    GROUP BY user_id
)
SELECT
    CASE
        WHEN num_interactions = 0 THEN 'No interactions'
        WHEN num_interactions BETWEEN 1 AND 2 THEN 'Cold (1-2)'
        WHEN num_interactions BETWEEN 3 AND 10 THEN 'Warm (3-10)'
        ELSE 'Active (10+)'
    END AS user_segment,
    COUNT(*) AS user_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) AS percentage
FROM user_interaction_counts
GROUP BY user_segment
ORDER BY user_count DESC;
```

**Expected finding:** Typically 30-40% of users are in the cold or no-interaction category. This is why cold start handling is not optional — it affects a substantial fraction of your user base.

### 4.2 Generate Training Labels (Implicit Feedback)

The hardest part of recommendation is defining what "positive" means when you have no explicit ratings.

**Design Decision: Use purchase as positive, view-without-purchase as negative.**
```sql
-- Create training pairs: (user_id, product_id, label)
CREATE OR REPLACE TABLE `your_project.ml_data.training_pairs` AS
SELECT
    e.user_id,
    CAST(REGEXP_EXTRACT(e.uri, r'/product/(\d+)') AS STRING) AS product_id,
    MAX(CASE WHEN e.event_type = 'purchase' THEN 1 ELSE 0 END) AS label,
    MIN(e.created_at) AS first_interaction_at,
    MAX(e.created_at) AS last_interaction_at,
    COUNT(*) AS num_interactions
FROM `bigquery-public-data.thelook_ecommerce.events` e
WHERE
    e.event_type IN ('product', 'purchase', 'cart')
    AND e.uri LIKE '/product/%'
    AND e.user_id IS NOT NULL
GROUP BY e.user_id, product_id
HAVING product_id IS NOT NULL;
```

**Why this labeling strategy?**
- Purchase = strong positive signal (user committed money)
- View-only = weak implicit positive (user was interested but didn't convert)
- Never-seen = implicit negative (but we don't know if they'd like it or just never saw it)

**Alternative considered:** Use clicks as positives, random items as negatives. This is simpler but introduces "false negatives" (items the user would like but never saw), degrading model quality.

### 4.3 Build User Feature Profiles

```sql
-- Aggregate user behavioral features for training
CREATE OR REPLACE TABLE `your_project.ml_data.user_features` AS
WITH user_stats AS (
    SELECT
        u.id AS user_id,
        u.age,
        u.gender,
        u.country,
        u.created_at AS user_created_at,
        COUNT(DISTINCT e.session_id) AS total_sessions,
        COUNT(e.id) AS total_events,
        COUNT(DISTINCT DATE(e.created_at)) AS active_days,
        SUM(CASE WHEN e.event_type = 'purchase' THEN 1 ELSE 0 END) AS total_purchases,
        MAX(e.created_at) AS last_active_at
    FROM `bigquery-public-data.thelook_ecommerce.users` u
    LEFT JOIN `bigquery-public-data.thelook_ecommerce.events` e
        ON u.id = e.user_id
    GROUP BY u.id, u.age, u.gender, u.country, u.created_at
)
SELECT
    user_id,
    age,
    gender,
    country,
    total_sessions,
    total_events,
    active_days,
    total_purchases,
    TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), last_active_at, DAY) AS days_since_last_active,
    TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), user_created_at, DAY) AS account_age_days,
    SAFE_DIVIDE(total_purchases, total_sessions) AS purchase_rate
FROM user_stats;
```

### 4.4 Build Event/Product Feature Profiles

```sql
-- Aggregate product (event) features for training
CREATE OR REPLACE TABLE `your_project.ml_data.event_features` AS
SELECT
    p.id AS product_id,
    p.name AS product_name,
    p.brand,
    p.category,
    p.department,
    p.retail_price,
    p.cost,
    (p.retail_price - p.cost) AS margin,
    COUNT(DISTINCT e.user_id) AS total_unique_viewers,
    COUNT(DISTINCT oi.user_id) AS total_unique_buyers,
    SAFE_DIVIDE(
        COUNT(DISTINCT oi.user_id),
        COUNT(DISTINCT e.user_id)
    ) AS conversion_rate
FROM `bigquery-public-data.thelook_ecommerce.products` p
LEFT JOIN `bigquery-public-data.thelook_ecommerce.events` e
    ON CAST(REGEXP_EXTRACT(e.uri, r'/product/(\d+)') AS INT64) = p.id
LEFT JOIN `bigquery-public-data.thelook_ecommerce.order_items` oi
    ON oi.product_id = p.id
GROUP BY p.id, p.name, p.brand, p.category, p.department, p.retail_price, p.cost;
```

---

## 5. Feature Engineering

With data explored, we can design the feature vectors that feed each tower of our model.

### 5.1 Feature Taxonomy

| Feature Type | User Tower | Event Tower |
|---|---|---|
| **Static** | Age, gender, country, account age | Category, price tier, duration |
| **Dynamic (batch)** | 30-day purchase rate, favorite categories | All-time click count, conversion rate |
| **Dynamic (real-time)** | Recent session events (last 10 clicks) | 5-min click count (Dataflow) |
| **Contextual** | Time of day, day of week, device type | Days until event, distance from user |

### 5.2 Feature Representation

The model ([model.py:32-55](model.py)) takes a **128-dimensional dense vector** as input to each tower. Here is how real-world features are packed into that vector:

**User feature vector (128 dims):**
```
Dims 0-15:    Age bucket one-hot (16 buckets, 5-year widths)
Dims 16-18:   Gender one-hot [M, F, Unknown]
Dims 19-50:   Country pseudo-embedding (32-dim, hash-based placeholder → learned Embedding in prod)
Dims 51-60:   Top-10 category purchase proportions (one-hot over favourite category)
Dims 61-70:   Activity features (log-scaled: sessions, events, active_days, purchases, purchase_rate)
Dims 71-80:   Recency features (log days-since-last-active, account age, recency decay)
Dims 81-127:  Recent click-sequence embedding (mean-pool of last 10 item hashes → GRU in prod)
```

**Event feature vector (128 dims):**
```
Dims 0-31:    Category pseudo-embedding (32-dim, hash-based → learned Embedding in prod)
Dims 32-39:   Price tier one-hot (8 buckets: <$10, $10-25, $25-50, $50-75, $75-100, $100-150, $150-200, $200+)
Dims 40-47:   Popularity features (log viewers, log buyers, conversion rate, log margin)
Dims 48-55:   Temporal features (log days-until, cyclic hour sin/cos, is_weekend, urgency decay)
Dims 56-63:   City/venue pseudo-embedding (8-dim, hash-based → learned Embedding in prod)
Dims 64-127:  Title bag-of-words embedding (mean-pool word hashes → textembedding-gecko@003 in prod)
```

This layout is implemented in [features.py](features.py) and used by `BigQueryInteractionDataset` via the `use_feature_encoders=True` flag ([model.py:163-232](model.py#L163)).

**Placeholder vs. production embeddings:**
The current implementation in `features.py` uses deterministic SHA-256 hash embeddings for string fields (country, category, city, title). These are runnable without a vocabulary file or pre-trained weights, but do not capture semantic similarity — "New York" and "Boston" hash to unrelated vectors. To upgrade:

| Field | Current (hash) | Production replacement |
|---|---|---|
| Country (32-dim) | `_hash_embedding` | `nn.Embedding(num_countries, 32)`, trained end-to-end |
| Category (32-dim) | `_hash_embedding` | `nn.Embedding(num_categories, 32)`, trained end-to-end |
| Click sequence (47-dim) | Mean-pool of hashes | Mean-pool real `EventTower` embeddings or GRU encoder |
| Title (64-dim) | Bag-of-word hashes | Vertex AI `textembedding-gecko@003` (768-dim → linear to 64) |

**Why dense vectors over sparse features?**
Sparse one-hot vectors for user IDs would require an embedding table with 10M entries that must be updated with every new user. By projecting all features into a dense input vector, we avoid that memory cost and generalize better to new users based on their behavioral profile.

### 5.3 Real-Time Feature Pipeline Deep Dive

The Dataflow pipeline ([dataflow_pipeline.py](dataflow_pipeline.py)) is the backbone of feature freshness.

**Why Pub/Sub → Dataflow → BigQuery (and not Kafka → Flink)?**
- We're on GCP. Managed Pub/Sub eliminates broker maintenance.
- Dataflow (Beam) auto-scales with incoming event volume — zero ops overhead.
- BigQuery as the Feature Store backing store enables both streaming writes AND analytical SQL queries on the same data.

**Understanding Sliding Windows ([dataflow_pipeline.py:57-58](dataflow_pipeline.py#L57)):**
```python
| 'Window' >> beam.WindowInto(window.SlidingWindows(size=5*60, period=1*60))
```

A sliding window of `size=300s, period=60s` means:
- At any moment, we compute the click count over the **last 5 minutes**.
- This count is **updated every 60 seconds**.
- At time T=10:05, the window covers events from 10:00 to 10:05.
- At time T=10:06, the window covers events from 10:01 to 10:06.

**Why sliding (not tumbling) windows?**
A tumbling window resets to zero every 5 minutes, creating a sawtooth pattern. A sliding window gives a smoother, more representative signal. For "is this event trending right now?", smooth is better.

**Trade-off:** Sliding windows emit more output records (one per period), increasing BigQuery write load. At 1M events × 1 update/minute, that's up to 1M writes/minute. This is acceptable with BigQuery's streaming insert API.

---

## 6. Model Architecture: Two-Tower Deep Dive

### 6.1 Why Two-Tower and Not Alternatives?

The book asks you to reason about alternatives. Here is the structured comparison:

| Architecture | Recall Quality | Inference Latency | Scalability | Cold Start |
|---|---|---|---|---|
| **Matrix Factorization** | Medium | Low (dot product) | Medium | Poor (needs retraining for new users/items) |
| **Item-based CF** | Medium | Medium (similarity lookup) | Low (O(N²) item comparisons) | OK for items |
| **DeepFM / Cross Network** | High | **High** (can't pre-compute) | **Poor** (must score all N items at runtime) | Medium |
| **Two-Tower (ours)** | High | **Low** (ANN over pre-computed item embeddings) | **High** (decouple user/item towers) | Handled separately |

**The key insight:** With 1M events, scoring every user-event pair at runtime with a cross-feature model requires 1M forward passes per request. At 5ms/pass, that's 83 minutes per request. Completely infeasible. Two-Tower pre-computes all event embeddings offline, reducing online inference to: 1 user tower pass + 1 ANN search.

### 6.2 Architecture Code Walkthrough

**UserTower ([model.py:32-43](model.py#L32)):**
```python
class UserTower(nn.Module):
    def __init__(self, embedding_dim=64):
        super(UserTower, self).__init__()
        self.fc1 = nn.Linear(128, 256)   # Project 128-dim features to 256-dim
        self.fc2 = nn.Linear(256, embedding_dim)  # Compress to 64-dim embedding

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.normalize(x, p=2, dim=1)  # L2 normalize for cosine similarity
```

**Why L2 normalize the output?** ([model.py:43](model.py#L43))

L2 normalization constrains all embeddings to lie on the unit hypersphere. This means:
- Dot product between two embeddings equals cosine similarity.
- All embeddings have the same magnitude; only direction varies.
- Vector Search can use dot product distance, which is the fastest ANN metric (no square root needed).

If you skip normalization, a user with more training examples will learn a larger magnitude embedding, making similarity scores inconsistent across the corpus.

**EventTower ([model.py:45-55](model.py#L45)):**
Same architecture as UserTower. The symmetry is intentional — we want user and event embeddings to live in the **same 64-dimensional space** so their dot product is meaningful.

**Why 64 dimensions?**
- 64 is empirically good for 1M items: enough expressiveness to capture fine-grained preferences, small enough that ANN search remains fast.
- At 128 dims, ANN recall improves by ~2% but index size doubles and search latency increases ~40%.
- At 32 dims, search is faster but model quality drops significantly for nuanced categories.

### 6.3 Loss Function: InfoNCE

**The contrastive loss ([model.py:71-87](model.py#L71)):**
```python
def contrastive_loss(user_embeds, event_embeds, temperature):
    batch_size = user_embeds.shape[0]
    # Similarity matrix: logits[i][j] = similarity(user_i, event_j)
    logits = torch.matmul(user_embeds, event_embeds.T) / temperature
    # Positive pairs are on the diagonal
    labels = torch.arange(batch_size).to(user_embeds.device)
    # Treat it as classification: user_i should "choose" event_i from the batch
    loss_u = F.cross_entropy(logits, labels)
    loss_e = F.cross_entropy(logits.T, labels)
    return (loss_u + loss_e) / 2
```

**Intuition — what is this loss doing?**

Imagine a batch of 256 (user, event) pairs. Each pair represents a real interaction (positive pair). The loss asks: for user_i, can the model identify its matched event_i from all 256 events in the batch? The 255 unmatched events are **in-batch negatives** — they appear as negative examples for free, with no sampling required.

This is the **InfoNCE** (Information Noise-Contrastive Estimation) loss, equivalent to a multi-class cross-entropy where the "correct class" for user_i is event_i.

**Why bidirectional loss (loss_u + loss_e)?**
`loss_u` says: "user should find its event." `loss_e` says: "event should find its user." Training both directions makes the embedding space more symmetric and generally improves recall quality by ~3-5%.

**The temperature parameter ([model.py:64](model.py#L64)):**
```python
self.temperature = nn.Parameter(torch.tensor(0.07))
```

Temperature `τ` controls the sharpness of the similarity distribution:
- Low `τ` (e.g., 0.07): The model focuses on learning to separate very similar items — hard negatives are penalized more. Can lead to instability with noisy data.
- High `τ` (e.g., 1.0): Softer distribution, easier to optimize but the model doesn't learn fine-grained distinctions.
- Making it a **learned parameter** (not a fixed hyperparameter) lets the model find the optimal value during training.

**Why in-batch negatives are a good fit here:**
With a batch of 2048, each positive pair gets 2047 negatives at zero sampling cost. For an event platform with power-law popularity (most clicks go to a few events), random events in a batch are usually genuinely irrelevant to any given user — making them valid negatives.

**Known failure mode — Popularity Bias:**
If event_j is very popular, it will appear as a negative for many (user_i, event_i) pairs in the batch even when user_i might genuinely like event_j. The model learns to push popular events away from all users, which hurts recommendations. See Section 15.2 for LogQ correction.

---

## 7. Training Pipeline

### 7.1 Data Loading Strategy

In production, the `generate_dummy_data` function ([model.py:236-240](model.py#L236)) is replaced by a BigQuery data loader. Here is the production pattern:

```python
from google.cloud import bigquery
import torch
from torch.utils.data import Dataset, DataLoader

class BigQueryInteractionDataset(Dataset):
    """Loads user-event interaction pairs from BigQuery for training."""
    
    def __init__(self, project_id: str, dataset: str, table: str, split: str = 'train'):
        client = bigquery.Client(project=project_id)
        # Temporal split: train on interactions before 2024-01-01, validate after
        cutoff = "2024-01-01" if split == "train" else "2024-02-01"
        operator = "<" if split == "train" else ">="
        
        query = f"""
        SELECT
            u.age_bucket,
            u.gender_encoded,
            u.purchase_rate,
            u.total_sessions_log,
            e.category_encoded,
            e.price_bucket,
            e.popularity_log,
            e.conversion_rate
        FROM `{project_id}.{dataset}.training_pairs` tp
        JOIN `{project_id}.{dataset}.user_features` u ON tp.user_id = u.user_id
        JOIN `{project_id}.{dataset}.event_features` e ON tp.product_id = e.product_id
        WHERE tp.label = 1  -- only positive pairs for contrastive training
        AND tp.first_interaction_at {operator} '{cutoff}'
        """
        
        df = client.query(query).to_dataframe()
        self.user_features = torch.tensor(df[USER_COLS].values, dtype=torch.float32)
        self.event_features = torch.tensor(df[EVENT_COLS].values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.user_features)
    
    def __getitem__(self, idx):
        return self.user_features[idx], self.event_features[idx]
```

**Why temporal split (not random split)?**
In recommendation, random splits leak future information into training. If a user clicked an event on day 5 and we randomly put that in training, the model may learn that specific event for that user — not generalizable patterns. Temporal splits simulate the real deployment scenario: train on past, evaluate on future.

### 7.2 Training Configuration

The training loop ([model.py:270-324](model.py#L270)) with production settings:

| Hyperparameter | Value | Why |
|---|---|---|
| `embedding_dim` | 64 | Balance of quality vs. ANN speed |
| `batch_size` | 2048 | More in-batch negatives → better contrastive signal |
| `lr` | 1e-3 (Adam) | Adam adapts LR per parameter; good for embeddings |
| `epochs` | 5 | Diminishing returns beyond 5 with this loss function |
| Temperature `τ` | Learned, init=0.07 | Matches SimCLR paper's recommendation for retrieval |

**Why large batch size is critical for contrastive learning:**
With batch_size=256, each positive pair gets 255 negatives.
With batch_size=2048, each positive pair gets 2047 negatives.
The increased diversity of negatives forces the model to learn finer-grained distinctions. This is well-documented: SimCLR, MoCo, and CLIP all show monotonically improving performance with batch size up to ~4096.

**Hardware recommendation:** Use an A100 GPU (available as Vertex AI A100 nodes). The matrix multiplication in InfoNCE (`user_embeds @ event_embeds.T`) is `(2048, 64) @ (64, 2048)` — a shape that saturates A100 tensor cores perfectly.

### 7.3 KFP Pipeline Orchestration

The Vertex AI Pipelines orchestration ([pipeline.py](pipeline.py)) ensures reproducibility and dependency management.

**Why KFP (Kubeflow Pipelines) over a shell script?**
- **Artifact lineage:** Each pipeline run records which model was trained on which data version.
- **Caching:** If data hasn't changed, KFP re-uses cached outputs, skipping expensive retraining.
- **Dependency enforcement:** The `deploy_task.after(train_task)` ([pipeline.py:90](pipeline.py#L90)) ensures the index is only rebuilt with the new model's embeddings, never the old ones.

The pipeline has two stages:
1. **`train_model_op`** ([pipeline.py:14-36](pipeline.py#L14)): Submits a Vertex AI Custom Training Job. Returns the GCS URI of the saved model.
2. **`deploy_index_op`** ([pipeline.py:38-56](pipeline.py#L38)): Loads the trained EventTower, generates all event embeddings, and deploys them to Vector Search.

---

## 8. Offline Evaluation

Before deploying, you must evaluate model quality on a held-out test set. Never deploy a model whose offline metrics you cannot measure.

### 8.1 Evaluation Metrics

**Recall@K:** Of all events a user actually interacted with in the test set, what fraction appears in the top-K recommendations?

```python
def recall_at_k(user_embeddings, event_embeddings, ground_truth, k=10):
    """
    user_embeddings: (N_users, dim)
    event_embeddings: (N_events, dim)
    ground_truth: dict {user_idx: set of positive event_idxs}
    """
    scores = user_embeddings @ event_embeddings.T  # (N_users, N_events)
    top_k_indices = scores.topk(k, dim=1).indices  # (N_users, K)
    
    recalls = []
    for user_idx, positive_indices in ground_truth.items():
        recommended = set(top_k_indices[user_idx].tolist())
        relevant = positive_indices
        recall = len(recommended & relevant) / len(relevant)
        recalls.append(recall)
    
    return sum(recalls) / len(recalls)
```

**NDCG@K (Normalized Discounted Cumulative Gain):** Rewards finding relevant events at higher ranks.

```python
import numpy as np

def ndcg_at_k(user_embeddings, event_embeddings, ground_truth, k=10):
    scores = user_embeddings @ event_embeddings.T
    top_k_indices = scores.topk(k, dim=1).indices
    
    ndcgs = []
    for user_idx, positive_indices in ground_truth.items():
        dcg = 0.0
        for rank, event_idx in enumerate(top_k_indices[user_idx].tolist()):
            if event_idx in positive_indices:
                dcg += 1.0 / np.log2(rank + 2)  # rank is 0-indexed
        
        # Ideal DCG: all positives ranked first
        ideal_hits = min(len(positive_indices), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
        
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    
    return sum(ndcgs) / len(ndcgs)
```

**Target offline metrics (production bar):**

| Metric | Minimum Threshold | Good | Excellent |
|---|---|---|---|
| Recall@10 | > 0.15 | > 0.25 | > 0.40 |
| NDCG@10 | > 0.12 | > 0.20 | > 0.35 |
| MRR | > 0.10 | > 0.18 | > 0.30 |

**Why these specific thresholds?**
They're derived from academic literature on retrieval models for similar-scale recommendation problems. The absolute values depend heavily on your data — establish a baseline with a simple popularity-based recommender first, then aim to beat it significantly.

### 8.2 Baseline Comparison

Always compare against a non-ML baseline before claiming success:

```python
def popularity_baseline_recall_at_k(events_by_popularity, ground_truth, k=10):
    """Recommend the same top-K most popular events to everyone."""
    top_k_popular = set(events_by_popularity[:k])
    recalls = []
    for user_idx, positive_indices in ground_truth.items():
        recall = len(top_k_popular & positive_indices) / len(positive_indices)
        recalls.append(recall)
    return sum(recalls) / len(recalls)
```

If your Two-Tower model doesn't significantly beat a popularity baseline, the training data quality, feature engineering, or loss function needs to be revisited before deployment.

---

## 9. Infrastructure Setup (GCP)

### Prerequisites

1. A GCP project with billing enabled.
2. `gcloud` CLI installed and authenticated.
3. Python 3.9+ installed locally.
4. Docker installed (for building the serving container).

### 9.1 Set Environment Variables

We use environment variables across all scripts for portability. These are the only values you need to change for your project.

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

### 9.2 Authenticate and Configure GCP

```bash
gcloud config set project $GCP_PROJECT_ID
gcloud auth application-default login
```

### 9.3 Enable Required APIs

```bash
gcloud services enable \
  dataflow.googleapis.com \
  aiplatform.googleapis.com \
  pubsub.googleapis.com \
  bigquery.googleapis.com \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com
```

### 9.4 Create GCS Buckets

```bash
# Model artifacts (training outputs, embeddings)
gsutil mb -l $GCP_REGION gs://$MODEL_BUCKET
gsutil mb -l $GCP_REGION gs://$EMBEDDING_BUCKET

# Dataflow temp storage (required by Beam runner)
gsutil mb -l $GCP_REGION gs://$MODEL_BUCKET/temp
```

### 9.5 Create BigQuery Dataset

```bash
bq mk --location=US feature_store
```

### 9.6 Set Up Vertex AI Feature Store

The Feature Store holds user features for low-latency serving. The BigQuery table written by Dataflow backs the Feature Store's online serving layer.

```bash
# Create the Feature Online Store (this is the online serving layer)
gcloud ai feature-online-stores create event_feature_store \
    --project=$GCP_PROJECT_ID \
    --region=$GCP_REGION \
    --bigtable-auto-scaling-min-node-count=1 \
    --bigtable-auto-scaling-max-node-count=3 \
    --bigtable-auto-scaling-cpu-utilization-target=50
```

**Why Bigtable-backed Feature Store for online serving?**
BigQuery is fast for analytics but has ~200ms latency for single-row lookups. Bigtable (via Vertex AI Feature Store) provides < 10ms P99 latency for online feature fetching — essential for our 150ms SLA. The Feature Store syncs from BigQuery to Bigtable automatically on a schedule you control.

### 9.7 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 10. Running the Dataflow Pipeline

The Dataflow pipeline ([dataflow_pipeline.py](dataflow_pipeline.py)) handles real-time feature generation. It is always running — start it before training so features are available at serving time.

### 10.1 Create Pub/Sub Infrastructure

```bash
gcloud pubsub topics create event-topic
gcloud pubsub subscriptions create event-sub \
    --topic=event-topic \
    --ack-deadline=60  # 60 seconds to process each message before redelivery
```

**Why 60-second ack deadline?**
Beam's GroupByKey operation may hold messages in flight across window boundaries. A short ack deadline (default 10s) causes Pub/Sub to redeliver messages that Beam is still processing, leading to duplicate writes. 60s gives sufficient buffer for window operations.

### 10.2 Run the Pipeline

**Local testing (DirectRunner — no GCP resources consumed):**
```bash
python dataflow_pipeline.py \
    --input_subscription=$PUBSUB_SUBSCRIPTION \
    --output_table=$BQ_FEATURE_TABLE
```

**Production deployment (DataflowRunner — fully managed, auto-scaling):**
```bash
python dataflow_pipeline.py \
    --input_subscription=$PUBSUB_SUBSCRIPTION \
    --output_table=$BQ_FEATURE_TABLE \
    --runner=DataflowRunner \
    --project=$GCP_PROJECT_ID \
    --region=$GCP_REGION \
    --temp_location=gs://$MODEL_BUCKET/temp \
    --max_num_workers=20 \
    --autoscaling_algorithm=THROUGHPUT_BASED
```

**Understanding the pipeline's data flow ([dataflow_pipeline.py:47-70](dataflow_pipeline.py#L47)):**

```
Pub/Sub Message (JSON bytes)
    │
    ▼ ParseEvent DoFn
    (event_id, 1)  -- key-value pair for aggregation
    │
    ▼ SlidingWindows(size=300s, period=60s)
    Groups events into 5-minute sliding windows
    │
    ▼ CombinePerKey(sum)
    (event_id, click_count_in_window)
    │
    ▼ FormatForBigQuery DoFn
    {"event_id": "...", "click_count_5m": N, "feature_timestamp": "..."}
    │
    ▼ WriteToBigQuery
    Appended to BigQuery feature table (backing Vertex AI Feature Store)
```

### 10.3 Simulate Pub/Sub Events for Testing

To test without a real event stream, publish synthetic messages:

```bash
# Publish 100 test events
for i in {1..100}; do
    gcloud pubsub topics publish event-topic \
        --message="{\"event_id\": \"event_$((RANDOM % 20))\", \"user_id\": \"user_$i\", \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"
done
```

**Verify data in BigQuery:**
```bash
bq query --use_legacy_sql=false \
    "SELECT event_id, click_count_5m, feature_timestamp
     FROM \`$GCP_PROJECT_ID.feature_store.event_features_rt\`
     ORDER BY feature_timestamp DESC
     LIMIT 20"
```

---

## 11. Running the Vertex AI Training Pipeline

The KFP pipeline ([pipeline.py](pipeline.py)) orchestrates training and index deployment.

### 11.1 Set Training Environment

```bash
export TRAINING_IMAGE="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13:latest"
```

**Why use Vertex AI's pre-built PyTorch image?**
Building a custom Docker image for training takes 10-15 minutes per iteration. Vertex AI's curated images have PyTorch, CUDA, cuDNN pre-installed and validated. Use a custom image only if you need dependencies not in the pre-built image.

### 11.2 Run the Pipeline

```bash
python pipeline.py
```

This compiles the pipeline to `recommendation_pipeline.json` and submits it. Monitor progress in GCP Console: **Vertex AI > Pipelines**.

**What happens during training ([model.py:270-324](model.py#L270)):**
1. Vertex AI allocates a GPU VM.
2. The Docker image is pulled and the training script starts.
3. Each epoch: `generate_dummy_data` (replace with BigQuery loader in production) → forward pass → InfoNCE loss → Adam gradient step.
4. After training: `save_model_to_gcs` uploads 3 artifacts:
   - `model.pth` — full model (user + event tower, for evaluation)
   - `event_tower.pth` — event tower only (for batch embedding generation)
   - `user_tower.pth` — user tower only (for serving)

### 11.3 Generate Embeddings and Deploy Vector Search Index

After training completes ([index_deploy.py](index_deploy.py)):

```bash
python index_deploy.py \
    --project_id=$GCP_PROJECT_ID \
    --region=$GCP_REGION \
    --gcs_bucket=$EMBEDDING_BUCKET \
    --gcs_prefix=vector_search_data \
    --num_events=1000000  # 1M events in production
```

**What this script does ([index_deploy.py:11-60](index_deploy.py#L11)):**
1. Loads the trained `EventTower` weights.
2. Runs all 1M event feature vectors through the EventTower → 64-dim embeddings.
3. Serializes embeddings as JSONL: `{"id": "event_123", "embedding": [0.1, -0.3, ...]}`.
4. Uploads the JSONL file to GCS.

**Then deploys the Vector Search index ([index_deploy.py:62-91](index_deploy.py#L62)):**
```python
my_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name=index_name,
    contents_delta_uri=gcs_uri,
    dimensions=64,
    approximate_neighbors_count=10,
    distance_measure_type="DOT_PRODUCT_DISTANCE",
)
```

**Why Tree-AH (ScaNN) and DOT_PRODUCT_DISTANCE?**
- **Tree-AH** is Google's ScaNN algorithm — it partitions the embedding space into a tree structure, enabling sub-linear search time (O(log N) vs O(N) for brute force).
- **DOT_PRODUCT_DISTANCE:** Because embeddings are L2-normalized, dot product equals cosine similarity. Google's benchmarks show dot product is 2-3x faster than Euclidean distance in ScaNN because it avoids the square root computation.

> **Note:** Vector Search index deployment takes 30-60 minutes for 1M items. This is expected — it's building and sharding the ANN index. Plan for this in your retraining schedule.

### 11.4 Handle Item Cold Start with Streaming Updates

When a new event is listed, use the streaming update API to add it immediately:

```python
from google.cloud import aiplatform

def add_new_event_to_index(event_id: str, event_features: list, index_name: str):
    """Adds a new event embedding to the live Vector Search index."""
    model = EventTower(embedding_dim=64)
    model.load_state_dict(torch.load('event_tower.pth'))
    model.eval()
    
    with torch.no_grad():
        features_tensor = torch.tensor([event_features], dtype=torch.float32)
        embedding = model(features_tensor).squeeze(0).tolist()
    
    index = aiplatform.MatchingEngineIndex(index_name=index_name)
    index.upsert_datapoints(
        datapoints=[{"datapoint_id": event_id, "feature_vector": embedding}]
    )
    print(f"Event {event_id} added to index in real-time.")
```

This satisfies our requirement: new events are recommendable within 10 minutes of creation.

---

## 12. Deploying the Serving Layer

The FastAPI serving layer ([serving.py](serving.py)) handles all online inference.

### 12.1 Capture the Vector Search Endpoint ID

After index deployment completes:

```bash
export INDEX_ENDPOINT_ID="your-deployed-index-endpoint-id"
export DEPLOYED_INDEX_ID="event_index_deployed"
```

**Find your endpoint ID:**
```bash
gcloud ai index-endpoints list \
    --project=$GCP_PROJECT_ID \
    --region=$GCP_REGION
```

### 12.2 Build and Deploy to Cloud Run

```bash
gcloud builds submit --tag gcr.io/$GCP_PROJECT_ID/serving-api

gcloud run deploy serving-api \
    --image gcr.io/$GCP_PROJECT_ID/serving-api \
    --platform managed \
    --region $GCP_REGION \
    --allow-unauthenticated \
    --min-instances=5 \
    --max-instances=200 \
    --concurrency=100 \
    --cpu=2 \
    --memory=2Gi \
    --set-env-vars GCP_PROJECT_ID=$GCP_PROJECT_ID,GCP_REGION=$GCP_REGION,INDEX_ENDPOINT_ID=$INDEX_ENDPOINT_ID,DEPLOYED_INDEX_ID=$DEPLOYED_INDEX_ID
```

**Why `--min-instances=5`?**
Cloud Run's default behavior scales to zero when idle. The first request after scale-to-zero triggers a cold start: loading the PyTorch model, initializing GCP clients — easily 3-10 seconds. Setting `min-instances=5` keeps at least 5 warm replicas ready, eliminating cold start latency for real users.

**Why `--concurrency=100`?**
Each serving replica can handle 100 concurrent requests. The Python GIL doesn't block FastAPI's async request handling because the heavy operations (Feature Store fetch, Vector Search query) are I/O-bound, not CPU-bound. 100 is the Cloud Run maximum, giving maximum throughput per instance.

### 12.3 Serving Architecture Deep Dive

**Request flow through `serving.py`:**

1. **Feature retrieval ([serving.py:106-122](serving.py#L106)):**
   ```python
   features = fetch_user_features(request.user_id)
   ```
   Fetches from Vertex AI Feature Store → returns 128-dim tensor.
   Target: < 10ms. If Feature Store is unreachable, falls back to demographic average.

2. **User tower inference ([serving.py:190-191](serving.py#L190)):**
   ```python
   with torch.no_grad():
       user_embedding = model(features).squeeze(0).tolist()
   ```
   PyTorch `no_grad()` disables gradient tracking, reducing memory by ~50% and speeding up inference. Target: < 5ms on CPU.

3. **Vector search ([serving.py:130-141](serving.py#L130)):**
   ```python
   response = index_endpoint.find_neighbors(
       deployed_index_id=DEPLOYED_INDEX_ID,
       queries=[embedding],
       num_neighbors=num_neighbors
   )
   ```
   ScaNN searches 1M event embeddings. Target: < 20ms.

**Why retry with exponential backoff ([serving.py:105](serving.py#L105) and [serving.py:129](serving.py#L129))?**
```python
@retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3))
```
Feature Store and Vector Search are remote services with occasional transient failures. Without retry, a single network hiccup causes a user-visible error. With exponential backoff (1s, 2s, 4s up to 10s), transient errors are silently recovered.

**Lazy initialization ([serving.py:93-98](serving.py#L93)):**
```python
@app.on_event("startup")
async def startup_event():
    load_model()  # pre-load PyTorch model on startup
    # GCP clients initialized lazily on first request
```
The PyTorch model is loaded at startup to eliminate per-request model loading overhead. GCP clients are initialized lazily — not at startup — to allow the container to start and pass health checks even without valid GCP credentials (useful for local testing).

### 12.4 Test the System

```bash
CLOUD_RUN_URL=$(gcloud run services describe serving-api \
    --region=$GCP_REGION \
    --format='value(status.url)')

curl -X POST $CLOUD_RUN_URL/recommend \
    -H "Content-Type: application/json" \
    -d '{"user_id": "user_123", "num_recommendations": 10}'
```

**Expected response:**
```json
{
    "user_id": "user_123",
    "recommendations": [
        {"id": "event_847", "distance": 0.923},
        {"id": "event_291", "distance": 0.901},
        ...
    ],
    "latency_seconds": 0.0847
}
```

**Load testing:**
```bash
# Install hey (HTTP load tester)
brew install hey

# Send 1000 requests with 50 concurrent connections
hey -n 1000 -c 50 -m POST \
    -H "Content-Type: application/json" \
    -d '{"user_id": "user_123", "num_recommendations": 10}' \
    $CLOUD_RUN_URL/recommend
```

Target: P50 < 80ms, P99 < 150ms, 0 errors.

---

## 13. Online Evaluation & A/B Testing

Offline metrics tell you the model is better in a held-out test. Online metrics tell you if it's better for real users spending real money. They often disagree — always A/B test before shipping.

### 13.1 A/B Testing Architecture

```
User Request
    │
    ▼
Traffic Splitter (Cloud Load Balancer Header-based routing)
    │
    ├──── 90% ──► Control (current model, e.g., v1)
    │
    └──── 10% ──► Treatment (new model, e.g., v2)
```

**Implementation using GCP Traffic Splitting (Cloud Run):**
```bash
# Deploy new model version without sending traffic to it
gcloud run deploy serving-api \
    --image gcr.io/$GCP_PROJECT_ID/serving-api:v2 \
    --no-traffic \
    --tag=v2

# Split 10% of traffic to the new version
gcloud run services update-traffic serving-api \
    --to-tags=v2=10 \
    --to-latest=90
```

### 13.2 Metrics to Track Online

| Metric | Measurement | Minimum Detectable Effect |
|---|---|---|
| Click-Through Rate (CTR) | Clicks / Impressions on recommendation widget | 0.5% absolute |
| Conversion Rate | Purchases / Recommendation clicks | 0.3% absolute |
| Coverage | Fraction of distinct events recommended | > 5% increase |
| Serendipity | Fraction of recommendations outside user's usual categories | Monitor for balance |

**Sample size calculation for statistical significance:**
For CTR baseline of 3%, MDE of 0.5%, power of 80%, alpha of 0.05 → you need approximately **50,000 users per variant** for a conclusive test. At 10% traffic split for the treatment with 50K MAU → test runs for ~10 days.

### 13.3 Log Recommendations for Analysis

Every recommendation made should be logged for evaluation:

```python
import json
from google.cloud import pubsub_v1

publisher = pubsub_v1.PublisherClient()

@app.post("/recommend")
async def get_recommendations(request: RecommendationRequest):
    # ... existing logic ...
    
    # Log for A/B analysis
    log_entry = {
        "user_id": request.user_id,
        "model_version": os.environ.get("MODEL_VERSION", "v1"),
        "recommendations": [r["id"] for r in recommendations],
        "timestamp": time.time(),
        "experiment_id": os.environ.get("EXPERIMENT_ID", "none"),
    }
    publisher.publish(
        "projects/{}/topics/recommendation-logs".format(PROJECT_ID),
        json.dumps(log_entry).encode()
    )
    
    return {"user_id": request.user_id, "recommendations": recommendations, ...}
```

Store click events from the UI with the same `experiment_id` and `model_version` to join recommendations with outcomes in BigQuery for analysis.

---

## 14. Monitoring & Observability

A deployed model degrades silently. Monitoring is what separates a production system from a demo.

### 14.1 Three Types of Drift to Monitor

**1. Data Drift (feature distribution shift):**
```sql
-- Monitor user feature distribution weekly
SELECT
    DATE_TRUNC(created_at, WEEK) AS week,
    AVG(age) AS avg_age,
    STDDEV(age) AS stddev_age,
    COUNT(DISTINCT user_id) AS unique_users
FROM `your_project.ml_data.user_features`
GROUP BY week
ORDER BY week DESC;
```

Alert if `avg_age` or `stddev_age` shifts by more than 2 standard deviations from the rolling 4-week average. This indicates the user population has changed and retraining is needed.

**2. Prediction Drift (output distribution shift):**
Monitor the distribution of similarity scores returned by Vector Search.
```python
# In serving.py, record score distributions
score_histogram = [r["distance"] for r in recommendations]
# Send to Cloud Monitoring as custom metric
```
If the average similarity score drops significantly, the user embedding space may have drifted away from the event embedding space (e.g., new event types that the model hasn't seen).

**3. Concept Drift (label distribution shift):**
Monitor CTR weekly. If CTR drops without a change in user behavior (as measured by total sessions), the model's recommendations are becoming less relevant.

### 14.2 Cloud Monitoring Dashboards

Key metrics to track in Cloud Monitoring:

```bash
# Create alerting policy for serving latency
gcloud alpha monitoring policies create \
    --notification-channels=YOUR_CHANNEL_ID \
    --display-name="Serving Latency P99 > 150ms" \
    --condition-display-name="p99 latency breached" \
    --condition-filter='resource.type="cloud_run_revision" AND metric.type="run.googleapis.com/request_latencies"' \
    --condition-threshold-value=150 \
    --condition-threshold-duration=300s
```

**Dashboard panels to build:**
1. Recommendation latency (P50, P95, P99) — by model version
2. Feature Store fetch latency — alert if > 15ms
3. Vector Search latency — alert if > 30ms
4. Error rate (5xx responses) — alert if > 0.1%
5. CTR trend (from BigQuery, updated hourly)
6. Feature freshness lag (time since last Dataflow write to BigQuery)

### 14.3 Retraining Triggers

| Trigger | Condition | Action |
|---|---|---|
| Scheduled | Every 24 hours | Full pipeline rerun (train + index rebuild) |
| CTR drop | CTR falls > 10% below 7-day average | Emergency retraining |
| Data drift alert | Feature distribution shifts > 2σ | Retrain + investigate root cause |
| New event burst | >10,000 new events in 1 hour | Trigger streaming upsert job only |

---

## 15. Trade-offs & Alternatives

### 15.1 Adding a Ranking Stage

**Current gap:** The Two-Tower model returns the top-K candidates by embedding similarity alone. Dot product between user and event embeddings captures general preference alignment but misses complex cross-feature interactions (e.g., "user prefers outdoor concerts in summer but indoor events in winter").

**Solution — Two-Stage Funnel:**

```
Two-Tower ANN (candidate generation)
    → top 500 candidates
    → DCN / LightGBM Ranker (re-ranks using cross features)
    → top 10 (served to user)
```

**Ranker Feature Engineering:**
The ranker can use features that the Two-Tower can't: the Hadamard product of the user and event embeddings (element-wise multiplication reveals which embedding dimensions are jointly active), plus raw contextual features like user's current location vs. event venue.

**Latency budget for ranking:**
- Candidate generation: ~30ms (Feature Store + UserTower + ANN)
- Ranking 500 candidates with DCN: ~20ms
- Total: ~50ms → well within 150ms SLA

### 15.2 LogQ Correction for Popularity Bias

**The problem:** In-batch negative sampling is biased toward popular events. If event_j appears in 10% of all user sessions, it appears as a negative example in ~10% of training batches. The model learns to push event_j away from most users, even when event_j is genuinely relevant.

**LogQ Correction:**
```python
def contrastive_loss_with_logq_correction(user_embeds, event_embeds, temperature, event_frequencies):
    """
    event_frequencies: tensor of shape (batch_size,), probability of each event
                       appearing in a random training batch.
    """
    logits = torch.matmul(user_embeds, event_embeds.T) / temperature
    
    # Subtract log probability of sampling each event as a negative
    # This debiases popular items
    log_q = torch.log(event_frequencies).unsqueeze(0)  # (1, batch_size)
    logits = logits - log_q
    
    labels = torch.arange(user_embeds.shape[0]).to(user_embeds.device)
    loss_u = F.cross_entropy(logits, labels)
    loss_e = F.cross_entropy(logits.T, labels)
    return (loss_u + loss_e) / 2
```

**Impact:** LogQ correction typically improves Recall@10 by 2-5% for long-tail events (events outside the top 5% by popularity) without degrading popular event recommendations.

### 15.3 Hard Negative Mining

**The problem:** Random in-batch negatives are too easy after a few epochs. The model easily distinguishes "jazz concert" from "comedy show" for a user whose history is 90% jazz. Easy negatives don't push the model to learn fine-grained distinctions.

**Solution — Hard Negatives:**
Mine negatives that are semantically close to positives but not interacted with:
```python
# After training epoch 2+, mine hard negatives
def mine_hard_negatives(user_embedding, event_embeddings, top_k_mine=100, exclude_positive=True):
    """
    Returns top-k similar events that the user did NOT interact with.
    These are harder negatives than random events.
    """
    similarities = user_embedding @ event_embeddings.T
    top_k = similarities.topk(top_k_mine + 1).indices  # +1 to exclude positive
    # Filter out the actual positive (which appears at top)
    hard_negatives = top_k[top_k != positive_event_idx][:top_k_mine]
    return hard_negatives
```

**Training schedule:** Train with random negatives for the first 2 epochs (to learn a reasonable embedding space), then switch to hard negatives. Starting with hard negatives from epoch 1 leads to training instability — the loss surface is too complex for the uninitialized model.

### 15.4 Asynchronous Feature Fetching

**Current state:** Features are fetched synchronously — the request waits for Feature Store before calling the UserTower ([serving.py:183-191](serving.py#L183)).

**Improvement:** Prefetch features in the background while the previous response is being sent. For sequential user requests (the same user making multiple requests), this eliminates the Feature Store latency almost entirely.

For a single request, use `asyncio` to parallelize Feature Store and any other I/O:
```python
import asyncio

@app.post("/recommend")
async def get_recommendations(request: RecommendationRequest):
    # Fetch user features and (hypothetically) contextual features in parallel
    user_features_task = asyncio.create_task(fetch_user_features_async(request.user_id))
    context_task = asyncio.create_task(fetch_context_async(request.user_id))
    
    user_features, context = await asyncio.gather(user_features_task, context_task)
    # ... rest of inference
```

Expected latency improvement: 10-30ms reduction in P50 latency when multiple feature sources are involved.

---

## Quick Reference: Architecture Decision Record

| Decision | Choice | Reason | Alternative |
|---|---|---|---|
| Feature streaming | Dataflow (Beam) | Managed, auto-scaling, native GCP | Kafka + Flink (more ops overhead) |
| Feature store | Vertex AI Feature Store | < 10ms online serving, BigQuery backed | Redis (faster but no backfill) |
| Model architecture | Two-Tower | Pre-computable event embeddings | DeepFM (better accuracy, infeasible latency) |
| Loss function | InfoNCE | Efficient use of in-batch negatives | BPR, binary cross-entropy |
| ANN algorithm | ScaNN (Tree-AH) | Best recall/latency tradeoff at 1M items | HNSW (better recall, higher memory) |
| Serving | Cloud Run (FastAPI) | Serverless, auto-scales to 0 | GKE (more control, more ops) |
| Orchestration | Vertex AI Pipelines (KFP) | Artifact lineage, caching, GCP native | Airflow (more flexible, more overhead) |
| Embedding dim | 64 | Good quality/speed tradeoff at 1M items | 128 (better quality, slower ANN) |

---

## Summary: What We Built

This system implements the industry-standard candidate generation architecture used at Pinterest, YouTube, Spotify, and Eventbrite:

1. **Real-time feature pipeline** (Dataflow) captures trending event signals with < 5-minute lag.
2. **Two-Tower model** (PyTorch + InfoNCE) learns user and event embeddings in a shared space.
3. **KFP pipeline** (Vertex AI) orchestrates reproducible training and index deployment.
4. **ANN index** (Vertex AI Vector Search / ScaNN) enables sub-20ms lookup across 1M events.
5. **FastAPI serving layer** (Cloud Run) handles 50K RPS with < 150ms P99 latency.
6. **Feature Store** (Vertex AI, Bigtable-backed) serves real-time user features in < 10ms.
7. **A/B testing framework** validates improvements before full rollout.
8. **Monitoring** catches drift, latency regressions, and CTR drops before they become outages.

The system is designed to evolve: add a ranking stage, implement hard negatives, or replace the Two-Tower with a session-based Transformer — each is a localized change that the modular architecture supports cleanly.
