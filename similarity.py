import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


WEIGHTS = {
    'artist':          0.25,
    'hourly':          0.15,
    'seasonality':     0.10,
    'day_of_week':     0.10,
    'listen_duration': 0.20,
    'engagement':      0.10,
    'skip_rate':       0.05,
    'diversity':       0.05,
}


def cosine_sim(vec1, vec2):
    """Cosine similarity between two dicts or arrays. Returns float in [0, 1]."""
    if isinstance(vec1, dict):
        keys = sorted(set(vec1) | set(vec2))
        v1 = np.array([vec1.get(k, 0.0) for k in keys])
        v2 = np.array([vec2.get(k, 0.0) for k in keys])
    else:
        v1 = np.array(vec1, dtype=float)
        v2 = np.array(vec2, dtype=float)

    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    return float(cosine_similarity([v1], [v2])[0][0])


def scalar_sim(a, b):
    """Similarity between two scalar values. Returns 1 - normalized absolute difference."""
    return 1.0 - abs(a - b)


def compute_final_score(scores, weights=WEIGHTS):
    """Weighted average — renormalizes weights to only present dimensions."""
    active = {k: weights[k] for k in scores if k in weights}
    total_weight = sum(active.values())
    return sum(active[k] * scores[k] for k in active) / total_weight if total_weight > 0 else 0.0


def print_report(scores, weights=WEIGHTS):
    final = compute_final_score(scores, weights)
    print("=" * 45)
    print(f"{'Dimension':<14} {'Score':>7}   {'Weight':>7}")
    print("-" * 45)
    for k, v in scores.items():
        print(f"  {k:<14} {v:>7.4f}   {weights.get(k, 0):>6.2f}")
    print("-" * 45)
    print(f"  {'Final':<14} {final:>7.4f}")
    print("=" * 45)
    return final
