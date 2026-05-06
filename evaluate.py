"""
Offline evaluation for the Two-Tower recommendation model.

Implements the metrics described in TUTORIAL.md Section 8:
  - Recall@K
  - NDCG@K  (Normalized Discounted Cumulative Gain)
  - MRR     (Mean Reciprocal Rank)
  - Popularity baseline (non-ML reference point)

Run against synthetic data to verify the evaluation harness:
    python evaluate.py

Run against a trained model checkpoint:
    python evaluate.py --model_path /tmp/model/model.pth
"""

import argparse
import os
import math
import torch
import torch.nn.functional as F

from model import TwoTowerModel, generate_dummy_data


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def recall_at_k(user_embeddings: torch.Tensor, event_embeddings: torch.Tensor,
                ground_truth: dict, k: int = 10) -> float:
    """
    Fraction of each user's positive events that appear in the top-K recommendations.

    Args:
        user_embeddings:  (N_users, dim) — one embedding per test user.
        event_embeddings: (N_events, dim) — embeddings for the full event corpus.
        ground_truth:     {user_idx: set of positive event indices}.
        k:                Number of recommendations to consider.

    Returns:
        Mean Recall@K across all users in ground_truth.
    """
    scores = user_embeddings @ event_embeddings.T  # (N_users, N_events)
    top_k = scores.topk(k, dim=1).indices         # (N_users, K)

    recalls = []
    for user_idx, positives in ground_truth.items():
        recommended = set(top_k[user_idx].tolist())
        recalls.append(len(recommended & positives) / len(positives))

    return sum(recalls) / len(recalls)


def ndcg_at_k(user_embeddings: torch.Tensor, event_embeddings: torch.Tensor,
              ground_truth: dict, k: int = 10) -> float:
    """
    NDCG@K rewards finding relevant events at higher ranks.

    Getting the positive at rank 1 scores log2(2)=1.0; at rank 10 only log2(11)≈0.29.
    Normalized by the ideal ordering (all positives ranked first).

    Args: same as recall_at_k.
    Returns: Mean NDCG@K across all users in ground_truth.
    """
    scores = user_embeddings @ event_embeddings.T
    top_k = scores.topk(k, dim=1).indices

    ndcgs = []
    for user_idx, positives in ground_truth.items():
        dcg = sum(
            1.0 / math.log2(rank + 2)
            for rank, event_idx in enumerate(top_k[user_idx].tolist())
            if event_idx in positives
        )
        ideal_hits = min(len(positives), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return sum(ndcgs) / len(ndcgs)


def mrr_at_k(user_embeddings: torch.Tensor, event_embeddings: torch.Tensor,
             ground_truth: dict, k: int = 10) -> float:
    """
    Mean Reciprocal Rank — the reciprocal of the rank of the first positive hit.

    Useful when you care about how quickly the model surfaces *any* relevant item.
    MRR is 1.0 if the positive is always ranked #1, and approaches 0 as it falls
    out of the top-K window.

    Args: same as recall_at_k.
    Returns: Mean MRR@K across all users in ground_truth.
    """
    scores = user_embeddings @ event_embeddings.T
    top_k = scores.topk(k, dim=1).indices

    rrs = []
    for user_idx, positives in ground_truth.items():
        rr = 0.0
        for rank, event_idx in enumerate(top_k[user_idx].tolist()):
            if event_idx in positives:
                rr = 1.0 / (rank + 1)
                break
        rrs.append(rr)

    return sum(rrs) / len(rrs)


def popularity_baseline_recall_at_k(event_popularity: list, ground_truth: dict,
                                    k: int = 10) -> float:
    """
    Non-ML baseline: recommend the same top-K most popular events to every user.

    If the Two-Tower model does not substantially beat this baseline, revisit
    feature engineering and training data quality before deploying.

    Args:
        event_popularity: List of event indices sorted by descending popularity.
        ground_truth:     {user_idx: set of positive event indices}.
        k:                Number of top-popular events to recommend.

    Returns:
        Mean Recall@K of the popularity baseline.
    """
    top_k_popular = set(event_popularity[:k])
    recalls = []
    for positives in ground_truth.values():
        recalls.append(len(top_k_popular & positives) / len(positives))
    return sum(recalls) / len(recalls)


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def build_synthetic_eval_set(model: TwoTowerModel, num_users: int = 200,
                              num_events: int = 5000, device: torch.device = torch.device('cpu')):
    """
    Constructs a synthetic evaluation set from the model's current weights.

    For each user we designate one event as their positive (the one they
    interacted with). In production this comes from the held-out test split
    of real interaction data.

    Returns:
        user_embeddings:  (num_users, dim)
        event_embeddings: (num_events, dim)
        ground_truth:     {user_idx: {positive_event_idx}}
        event_popularity: list of event indices in descending popularity order
                          (simulated via uniform random — all equally popular).
    """
    model.eval()

    # Generate user features and get embeddings
    user_feats = torch.randn(num_users, 128).to(device)
    with torch.no_grad():
        user_embeddings = model.user_tower(user_feats)

    # Generate event features and get embeddings
    event_feats = torch.randn(num_events, 128).to(device)
    with torch.no_grad():
        event_embeddings = model.event_tower(event_feats)

    # Assign one positive event per user (simulated interaction)
    ground_truth = {
        user_idx: {user_idx % num_events}  # deterministic positive for reproducibility
        for user_idx in range(num_users)
    }

    # Simulated popularity — in production compute from interaction counts
    import random
    event_popularity = list(range(num_events))
    random.shuffle(event_popularity)

    return user_embeddings, event_embeddings, ground_truth, event_popularity


def run_evaluation(model: TwoTowerModel, num_users: int = 200, num_events: int = 5000,
                   k: int = 10, device: torch.device = torch.device('cpu')) -> dict:
    """
    Runs all metrics against a synthetic evaluation set and prints a report.

    Returns a dict of metric_name → value for programmatic use.
    """
    user_embs, event_embs, ground_truth, event_popularity = build_synthetic_eval_set(
        model, num_users, num_events, device
    )

    results = {
        f'Recall@{k}':             recall_at_k(user_embs, event_embs, ground_truth, k),
        f'NDCG@{k}':               ndcg_at_k(user_embs, event_embs, ground_truth, k),
        f'MRR@{k}':                mrr_at_k(user_embs, event_embs, ground_truth, k),
        f'Popularity Recall@{k}':  popularity_baseline_recall_at_k(event_popularity, ground_truth, k),
    }

    print("\n" + "=" * 52)
    print(f"  Offline Evaluation  |  {num_users} users  |  {num_events} events  |  K={k}")
    print("=" * 52)
    for metric, value in results.items():
        bar = "#" * int(value * 40)
        label = "  [baseline]" if "Popularity" in metric else ""
        print(f"  {metric:<22} {value:.4f}  {bar}{label}")
    print("=" * 52)

    model_recall = results[f'Recall@{k}']
    baseline_recall = results[f'Popularity Recall@{k}']
    lift = (model_recall - baseline_recall) / max(baseline_recall, 1e-9)
    print(f"\n  Lift over popularity baseline: {lift:+.1%}")
    if lift < 0.05:
        print("  WARNING: Model barely beats the popularity baseline.")
        print("  Review feature engineering and training data before deploying.")
    else:
        print("  Model meaningfully outperforms the popularity baseline. Good to proceed.")
    print()

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Offline evaluation for the Two-Tower model.")
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to a saved model.pth checkpoint. If omitted, uses random weights.')
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--num_users', type=int, default=200,
                        help='Number of test users to evaluate against.')
    parser.add_argument('--num_events', type=int, default=5000,
                        help='Size of the event corpus to search over.')
    parser.add_argument('--k', type=int, default=10,
                        help='Cut-off rank for Recall@K, NDCG@K, MRR@K.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = TwoTowerModel(embedding_dim=args.embedding_dim).to(device)

    if args.model_path and os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model weights from '{args.model_path}'.")
    else:
        print("No model checkpoint found — evaluating with random (untrained) weights.")
        print("Train first with: python model.py --epochs 5")

    run_evaluation(model, num_users=args.num_users, num_events=args.num_events,
                   k=args.k, device=device)
