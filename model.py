import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from google.cloud import storage

try:
    from google.cloud import bigquery
    _BQ_AVAILABLE = True
except ImportError:
    _BQ_AVAILABLE = False

# Numeric columns loaded directly from BigQuery feature tables.
# In production, categorical fields (gender, country, category) are one-hot
# or embedding-encoded by a preprocessing layer before concatenation,
# bringing the total to the model's input_dim (128).
USER_COLS = [
    'age', 'total_sessions', 'total_events', 'active_days',
    'total_purchases', 'days_since_last_active', 'account_age_days', 'purchase_rate',
]
EVENT_COLS = [
    'retail_price', 'cost', 'margin',
    'total_unique_viewers', 'total_unique_buyers', 'conversion_rate',
]

# --- MODEL DEFINITION ---

class UserTower(nn.Module):
    def __init__(self, embedding_dim=64):
        super(UserTower, self).__init__()
        # In a real scenario, we would use Embedding layers for categorical features.
        # Here we simulate an input feature vector of size 128
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, embedding_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.normalize(x, p=2, dim=1)

class EventTower(nn.Module):
    def __init__(self, embedding_dim=64):
        super(EventTower, self).__init__()
        # Simulated event feature vector of size 128
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, embedding_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.normalize(x, p=2, dim=1)

class TwoTowerModel(nn.Module):
    def __init__(self, embedding_dim=64):
        super(TwoTowerModel, self).__init__()
        self.user_tower = UserTower(embedding_dim)
        self.event_tower = EventTower(embedding_dim)
        
        # Temperature parameter for InfoNCE loss
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, user_features, event_features):
        user_embeds = self.user_tower(user_features)
        event_embeds = self.event_tower(event_features)
        return user_embeds, event_embeds

def contrastive_loss(user_embeds, event_embeds, temperature):
    """
    Computes InfoNCE Loss (Normalized Temperature-scaled Cross Entropy).
    Assumes positive pairs are aligned in the batch (i.e., diagonal).
    """
    batch_size = user_embeds.shape[0]

    # Compute similarity matrix (batch_size x batch_size)
    logits = torch.matmul(user_embeds, event_embeds.T) / temperature

    # Labels are the diagonal (each user matches with its corresponding event)
    labels = torch.arange(batch_size).to(user_embeds.device)

    loss_u = F.cross_entropy(logits, labels)
    loss_e = F.cross_entropy(logits.T, labels)

    return (loss_u + loss_e) / 2


def contrastive_loss_with_logq_correction(user_embeds, event_embeds, temperature, event_frequencies):
    """
    InfoNCE with LogQ correction to debias popular events.

    In-batch negative sampling over-penalises popular events because they appear
    as negatives for many users. LogQ correction subtracts the log-probability of
    each event being sampled, neutralising the popularity bias.

    Args:
        event_frequencies: Tensor (batch_size,) — probability of each event
                           appearing in a random training batch, estimated from
                           global interaction counts.
    """
    logits = torch.matmul(user_embeds, event_embeds.T) / temperature

    # Subtract log-sampling probability for each event in the batch
    log_q = torch.log(event_frequencies.clamp(min=1e-9)).unsqueeze(0)  # (1, batch_size)
    logits = logits - log_q

    labels = torch.arange(user_embeds.shape[0]).to(user_embeds.device)
    loss_u = F.cross_entropy(logits, labels)
    loss_e = F.cross_entropy(logits.T, labels)
    return (loss_u + loss_e) / 2


def mine_hard_negatives(user_embedding, event_embeddings, positive_event_idx, top_k_mine=100):
    """
    Returns indices of the top-k most similar events that the user did NOT interact with.

    Call this after at least 2 epochs of training with random negatives so the
    embedding space is already meaningful. Starting with hard negatives on epoch 1
    causes instability because random embeddings produce misleading hard negatives.

    Args:
        user_embedding:    Tensor (dim,) — a single user embedding.
        event_embeddings:  Tensor (N_events, dim) — all event embeddings.
        positive_event_idx: int — index of the user's actual positive event (excluded).
        top_k_mine:        Number of hard negatives to return.

    Returns:
        Tensor of shape (top_k_mine,) containing event indices.
    """
    similarities = user_embedding @ event_embeddings.T  # (N_events,)
    # Retrieve top_k + 1 to ensure we can exclude the positive
    top_indices = similarities.topk(top_k_mine + 1).indices
    hard_negatives = top_indices[top_indices != positive_event_idx][:top_k_mine]
    return hard_negatives


# --- DATA LOADING ---

def _pad_to_dim(arr: np.ndarray, target_dim: int = 128) -> np.ndarray:
    """Pads feature array to target_dim columns. In production replace with proper preprocessing."""
    n, d = arr.shape
    if d < target_dim:
        arr = np.concatenate([arr, np.zeros((n, target_dim - d), dtype='float32')], axis=1)
    return arr[:, :target_dim]


class BigQueryInteractionDataset(Dataset):
    """
    Loads positive user-event interaction pairs from BigQuery for contrastive training.

    Requires the feature tables created by the SQL in TUTORIAL.md Section 4:
      - {project}.{dataset}.training_pairs
      - {project}.{dataset}.user_features
      - {project}.{dataset}.event_features

    The split is temporal, not random, to prevent future leakage:
      train  — interactions before train_cutoff
      val    — interactions between train_cutoff and val_cutoff
    """

    def __init__(
        self,
        project_id: str,
        dataset: str,
        split: str = 'train',
        train_cutoff: str = '2024-01-01',
        val_cutoff: str = '2024-02-01',
        input_dim: int = 128,
        use_feature_encoders: bool = False,
    ):
        """
        Args:
            use_feature_encoders: When True, applies UserFeatureEncoder /
                EventFeatureEncoder from features.py to produce the full
                128-dim vectors described in TUTORIAL.md Section 5.2.
                When False (default), raw BQ columns are zero-padded to
                input_dim — faster for quick experiments.
        """
        if not _BQ_AVAILABLE:
            raise ImportError("google-cloud-bigquery is required. Run: pip install google-cloud-bigquery")

        client = bigquery.Client(project=project_id)

        if split == 'train':
            time_filter = f"tp.first_interaction_at < '{train_cutoff}'"
        elif split == 'val':
            time_filter = (
                f"tp.first_interaction_at >= '{train_cutoff}' "
                f"AND tp.first_interaction_at < '{val_cutoff}'"
            )
        else:
            time_filter = f"tp.first_interaction_at >= '{val_cutoff}'"

        user_cols = ', '.join(f'u.{c}' for c in USER_COLS)
        event_cols = ', '.join(f'e.{c}' for c in EVENT_COLS)

        query = f"""
        SELECT
            {user_cols},
            {event_cols}
        FROM `{project_id}.{dataset}.training_pairs` tp
        JOIN `{project_id}.{dataset}.user_features` u USING (user_id)
        JOIN `{project_id}.{dataset}.event_features` e ON tp.product_id = e.product_id
        WHERE tp.label = 1
          AND {time_filter}
        """

        df = client.query(query).to_dataframe()
        df = df.fillna(0)

        if use_feature_encoders:
            from features import UserFeatureEncoder, EventFeatureEncoder
            user_enc = UserFeatureEncoder()
            event_enc = EventFeatureEncoder()
            user_vecs = np.stack([user_enc.encode(row) for row in df[USER_COLS].to_dict('records')])
            event_vecs = np.stack([event_enc.encode(row) for row in df[EVENT_COLS].to_dict('records')])
            self.user_features = torch.tensor(user_vecs)
            self.event_features = torch.tensor(event_vecs)
        else:
            # Zero-pad raw BQ columns to input_dim. Quick path for experiments.
            # Switch to use_feature_encoders=True for the full feature pipeline.
            user_arr = df[USER_COLS].values.astype('float32')
            event_arr = df[EVENT_COLS].values.astype('float32')
            self.user_features = torch.tensor(_pad_to_dim(user_arr, input_dim))
            self.event_features = torch.tensor(_pad_to_dim(event_arr, input_dim))

    def __len__(self) -> int:
        return len(self.user_features)

    def __getitem__(self, idx):
        return self.user_features[idx], self.event_features[idx]


def generate_dummy_data(batch_size=256):
    """Simulates loading data from BigQuery/GCS. Used when --use_bq_data is not set."""
    user_features = torch.randn(batch_size, 128)
    event_features = torch.randn(batch_size, 128)
    return user_features, event_features

def save_model_to_gcs(model, bucket_name, prefix):
    """Saves PyTorch state dicts locally and uploads to GCS."""
    os.makedirs('/tmp/model', exist_ok=True)
    
    full_model_path = '/tmp/model/model.pth'
    event_tower_path = '/tmp/model/event_tower.pth'
    user_tower_path = '/tmp/model/user_tower.pth'
    
    torch.save(model.state_dict(), full_model_path)
    torch.save(model.event_tower.state_dict(), event_tower_path)
    torch.save(model.user_tower.state_dict(), user_tower_path)
    
    if not bucket_name:
        print("No GCS bucket provided, models saved to /tmp/model/")
        return

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    for path, name in [
        (full_model_path, f'{prefix}/model.pth'),
        (event_tower_path, f'{prefix}/event_tower.pth'),
        (user_tower_path, f'{prefix}/user_tower.pth')
    ]:
        blob = bucket.blob(name)
        blob.upload_from_filename(path)
        print(f"Uploaded {name} to gs://{bucket_name}/{prefix}/")

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = TwoTowerModel(embedding_dim=args.embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --- Data source selection ---
    if args.use_bq_data:
        print(f"Loading training data from BigQuery: {args.bq_project}.{args.bq_dataset}")
        train_dataset = BigQueryInteractionDataset(
            project_id=args.bq_project,
            dataset=args.bq_dataset,
            split='train',
            input_dim=128,
        )
        loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        use_loader = True
        print(f"Loaded {len(train_dataset):,} training pairs from BigQuery.")
    else:
        print("Using synthetic dummy data (pass --use_bq_data for production BigQuery data).")
        use_loader = False

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        steps = 0

        if use_loader:
            for user_features, event_features in loader:
                user_features = user_features.to(device)
                event_features = event_features.to(device)
                optimizer.zero_grad()
                user_embeds, event_embeds = model(user_features, event_features)
                loss = contrastive_loss(user_embeds, event_embeds, model.temperature)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                steps += 1
        else:
            for _ in range(50):
                user_features, event_features = generate_dummy_data(args.batch_size)
                user_features = user_features.to(device)
                event_features = event_features.to(device)
                optimizer.zero_grad()
                user_embeds, event_embeds = model(user_features, event_features)
                loss = contrastive_loss(user_embeds, event_embeds, model.temperature)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                steps += 1

        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {epoch_loss/steps:.4f}")

    print("Training complete.")
    save_model_to_gcs(model, args.gcs_bucket, args.gcs_prefix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--gcs_bucket', type=str, default=os.environ.get('MODEL_BUCKET', 'YOUR_BUCKET'))
    parser.add_argument('--gcs_prefix', type=str, default=os.environ.get('MODEL_PREFIX', 'recommendation_model'))

    # BigQuery data loading (production path)
    parser.add_argument('--use_bq_data', action='store_true',
                        help='Load training data from BigQuery instead of synthetic dummy data.')
    parser.add_argument('--bq_project', type=str, default=os.environ.get('GCP_PROJECT_ID', 'YOUR_PROJECT_ID'))
    parser.add_argument('--bq_dataset', type=str, default='ml_data',
                        help='BigQuery dataset containing training_pairs, user_features, event_features tables.')

    args = parser.parse_args()
    train(args)
