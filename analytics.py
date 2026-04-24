import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score

from data import p1, p2
from dimensions import (
    hourly_vector, artist_vector, seasonal_vector, dow_vector,
    skip_rate, track_diversity, listen_duration_vector, engagement_vector,
)
from similarity import cosine_sim, scalar_sim, compute_final_score, WEIGHTS  # noqa: F401 (re-exported)
from clustering import build_artist_vocab, build_track_raw_features


def _adapt(df):
    """Add minutesPlayed alias required by dimensions.py."""
    d = df.copy()
    if 'minutesPlayed' not in d.columns:
        d['minutesPlayed'] = d['minPlayed']
    return d


def _run_similarity():
    m, a = _adapt(p1), _adapt(p2)
    scores = {
        'artist':          cosine_sim(artist_vector(m),          artist_vector(a)),
        'hourly':          cosine_sim(hourly_vector(m),          hourly_vector(a)),
        'seasonality':     cosine_sim(seasonal_vector(m),        seasonal_vector(a)),
        'day_of_week':     cosine_sim(dow_vector(m),             dow_vector(a)),
        'listen_duration': cosine_sim(listen_duration_vector(m), listen_duration_vector(a)),
        'engagement':      cosine_sim(engagement_vector(m),      engagement_vector(a)),
        'skip_rate':       scalar_sim(skip_rate(m),              skip_rate(a)),
        'diversity':       scalar_sim(track_diversity(m),        track_diversity(a)),
    }
    return scores, compute_final_score(scores)


def _run_clustering(n_clusters=2, random_state=42):
    m, a = _adapt(p1), _adapt(p2)

    artist_vocab = build_artist_vocab(m, a)
    rows_me = build_track_raw_features(m, user_label=0, artist_vocab=artist_vocab)
    rows_at = build_track_raw_features(a, user_label=1, artist_vocab=artist_vocab)
    all_rows = rows_me + rows_at

    X = np.vstack([r['features'] for r in all_rows])
    true_labels = np.array([r['user'] for r in all_rows])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = min(50, X_scaled.shape[1] - 1)
    pca_pre = PCA(n_components=n_components, random_state=random_state)
    X_reduced = pca_pre.fit_transform(X_scaled)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = km.fit_predict(X_reduced)

    ari = adjusted_rand_score(true_labels, cluster_labels)
    sil = silhouette_score(X_reduced, cluster_labels)

    pca2 = PCA(n_components=2, random_state=random_state)
    X_2d = pca2.fit_transform(X_reduced)
    var = pca2.explained_variance_ratio_

    distances = km.transform(X_reduced).min(axis=1)
    outliers = pd.DataFrame({
        'track':   [r['track']  for r in all_rows],
        'artist':  [r['artist'] for r in all_rows],
        'user':    ['Person 1' if r['user'] == 0 else 'Person 2' for r in all_rows],
        'distance': distances,
        'cluster': cluster_labels,
    }).sort_values('distance', ascending=False).reset_index(drop=True)

    cluster_stats = []
    for c in range(n_clusters):
        mask = cluster_labels == c
        total = int(mask.sum())
        me_cnt = int(((true_labels == 0) & mask).sum())
        at_cnt = int(((true_labels == 1) & mask).sum())
        dominant = 'Person 1' if me_cnt >= at_cnt else 'Person 2'
        cluster_stats.append({
            'cluster': c, 'total': total,
            'person1': me_cnt, 'person2': at_cnt,
            'dominant': dominant,
            'purity': round(max(me_cnt, at_cnt) / total * 100, 1),
        })

    return {
        'X_2d': X_2d, 'var': var,
        'cluster_labels': cluster_labels, 'true_labels': true_labels,
        'ari': ari, 'sil': sil,
        'outliers': outliers, 'cluster_stats': cluster_stats,
    }


print("Computing similarity & clustering analytics…")
SIM_SCORES, SIM_FINAL = _run_similarity()
CLUSTER_DATA = _run_clustering()
