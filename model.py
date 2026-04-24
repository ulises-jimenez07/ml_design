import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from google.cloud import storage

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

# --- TRAINING LOOP ---

def generate_dummy_data(batch_size=256):
    """Simulates loading data from BigQuery/GCS"""
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
    
    # Simulated Training Loop
    epochs = args.epochs
    steps_per_epoch = 50
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            user_features, event_features = generate_dummy_data(args.batch_size)
            user_features = user_features.to(device)
            event_features = event_features.to(device)
            
            optimizer.zero_grad()
            user_embeds, event_embeds = model(user_features, event_features)
            
            loss = contrastive_loss(user_embeds, event_embeds, model.temperature)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/steps_per_epoch:.4f}")
        
    print("Training complete.")
    
    # Save artifacts
    save_model_to_gcs(model, args.gcs_bucket, args.gcs_prefix)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # AIP_MODEL_DIR is provided by Vertex AI Training by default
    default_gcs_path = os.environ.get('AIP_MODEL_DIR', 'gs://YOUR_BUCKET/models/')
    
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--gcs_bucket', type=str, default=os.environ.get('MODEL_BUCKET', 'YOUR_BUCKET'), help='GCS Bucket for saving model')
    parser.add_argument('--gcs_prefix', type=str, default=os.environ.get('MODEL_PREFIX', 'recommendation_model'), help='Prefix path in GCS')
    
    args = parser.parse_args()
    train(args)
