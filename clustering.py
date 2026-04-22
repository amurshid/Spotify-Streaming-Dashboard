import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score

from dimensions import (
    track_hourly_vector, track_seasonal_vector, track_dow_vector,
    track_listen_duration_vector, track_engagement_vector, track_skip_rate,
)


def build_artist_vocab(df_me, df_at):
    """Sorted dict of all unique artists across both users → index."""
    artists = set(df_me['artistName'].unique()) | set(df_at['artistName'].unique())
    return {a: i for i, a in enumerate(sorted(artists))}


def build_track_raw_features(df, user_label, artist_vocab):
    """
    One row per unique (trackName, artistName).
    Features: behavioral (46-dim) + artist one-hot (N-dim).
    Artist one-hot encodes WHICH artist the track belongs to — the one
    dimension where the two users are nearly 0 similar (score 0.048).
    """
    rows = []
    for (track, artist), group in df.groupby(['trackName', 'artistName']):
        behavioral = np.concatenate([
            track_hourly_vector(group),          # 24
            track_seasonal_vector(group),        # 4
            track_dow_vector(group),             # 7
            track_listen_duration_vector(group), # 6
            track_engagement_vector(group),      # 3
            [track_skip_rate(group)],            # 1
            [len(group)],                        # 1 (play count)
        ])
        artist_vec = np.zeros(len(artist_vocab))
        if artist in artist_vocab:
            artist_vec[artist_vocab[artist]] = 1.0

        rows.append({
            'track': track,
            'artist': artist,
            'user': user_label,
            'features': np.concatenate([behavioral, artist_vec]),
        })
    return rows


def run_kmeans_clustering(df_me, df_at, n_clusters=2, random_state=42):
    print("\n--- K-Means Clustering (k=2, artist-based) ---\n")

    artist_vocab = build_artist_vocab(df_me, df_at)
    print(f"Artist vocabulary size : {len(artist_vocab)} unique artists")

    rows_me = build_track_raw_features(df_me, user_label=0, artist_vocab=artist_vocab)
    rows_at = build_track_raw_features(df_at, user_label=1, artist_vocab=artist_vocab)
    all_rows = rows_me + rows_at

    X = np.vstack([r['features'] for r in all_rows])
    true_labels = np.array([r['user'] for r in all_rows])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reduce before clustering: high-dim sparse artist one-hot breaks euclidean
    # distance (curse of dimensionality). PCA compresses artist structure into
    # dense components that K-Means can actually use.
    n_components = min(50, X_scaled.shape[1] - 1)
    pca_pre = PCA(n_components=n_components, random_state=random_state)
    X_reduced = pca_pre.fit_transform(X_scaled)
    explained = pca_pre.explained_variance_ratio_.sum()
    print(f"PCA pre-reduction     : {n_components} components, {explained:.1%} variance retained")

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = km.fit_predict(X_reduced)

    ari = adjusted_rand_score(true_labels, cluster_labels)
    sil = silhouette_score(X_reduced, cluster_labels)

    print(f"Tracks in dataset   : {len(all_rows):,}  (you: {len(rows_me)}, Atharva: {len(rows_at)})")
    print(f"Adjusted Rand Index : {ari:.4f}  (1.0 = perfect separation, 0.0 = random)")
    print(f"Silhouette Score    : {sil:.4f}  (higher = more distinct clusters)")

    _print_cluster_purity(cluster_labels, true_labels, n_clusters)
    _detect_outliers(km, X_reduced, all_rows, top_n=10)
    _plot_pca(X_reduced, cluster_labels, true_labels, ari)

    return cluster_labels, true_labels, ari


def _detect_outliers(km, X_reduced, all_rows, top_n=10):
    """
    Outlier score = distance from each track to its assigned cluster centroid.
    Tracks farthest from their centroid have the most unusual listening patterns.
    """
    # km.transform returns distance to every centroid; min = distance to own cluster
    distances = km.transform(X_reduced).min(axis=1)

    df = pd.DataFrame({
        'track':    [r['track']  for r in all_rows],
        'artist':   [r['artist'] for r in all_rows],
        'user':     ['You' if r['user'] == 0 else 'Atharva' for r in all_rows],
        'distance': distances,
    }).sort_values('distance', ascending=False)

    def safe(s):
        return s.encode('ascii', 'replace').decode('ascii')

    print(f"\n--- Outlier Detection: Top {top_n} most unusual tracks (farthest from centroid) ---\n")
    print(f"{'Rank':<5} {'User':<10} {'Distance':>9}  Track - Artist")
    print("-" * 70)
    for rank, (_, row) in enumerate(df.head(top_n).iterrows(), start=1):
        print(f"{rank:<5} {row['user']:<10} {row['distance']:>9.4f}  {safe(row['track'])} - {safe(row['artist'])}")

    print(f"\n--- Top {top_n} outliers per user ---\n")
    for user_name in ['You', 'Atharva']:
        print(f"  {user_name}:")
        subset = df[df['user'] == user_name].head(top_n)
        for _, row in subset.iterrows():
            print(f"    {row['distance']:.4f}  {safe(row['track'])} - {safe(row['artist'])}")
        print()


def _print_cluster_purity(cluster_labels, true_labels, n_clusters):
    print("\nCluster composition:")
    for c in range(n_clusters):
        mask = cluster_labels == c
        total = mask.sum()
        me_count = ((true_labels == 0) & mask).sum()
        at_count = ((true_labels == 1) & mask).sum()
        dominant = "You" if me_count >= at_count else "Atharva"
        purity = max(me_count, at_count) / total * 100
        print(f"  Cluster {c}: {total:4d} tracks  |  You: {me_count:4d}  Atharva: {at_count:4d}"
              f"  → mostly {dominant} ({purity:.1f}% pure)")


def _plot_pca(X_reduced, cluster_labels, true_labels, ari):
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_reduced)
    var = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for c, marker, label in [(0, 'o', 'Cluster 0'), (1, 's', 'Cluster 1')]:
        mask = cluster_labels == c
        axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1],
                        marker=marker, alpha=0.45, s=18, label=label)
    axes[0].set_title(f'K-Means clusters (ARI = {ari:.3f})')
    axes[0].set_xlabel(f'PC1 ({var[0]:.1%} var)')
    axes[0].set_ylabel(f'PC2 ({var[1]:.1%} var)')
    axes[0].legend()

    colors = {0: '#1f77b4', 1: '#ff7f0e'}
    user_names = {0: 'You', 1: 'Atharva'}
    for u in [0, 1]:
        mask = true_labels == u
        axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1],
                        c=colors[u], alpha=0.45, s=18, label=user_names[u])
    axes[1].set_title('True user identity')
    axes[1].set_xlabel(f'PC1 ({var[0]:.1%} var)')
    axes[1].set_ylabel(f'PC2 ({var[1]:.1%} var)')
    axes[1].legend()

    plt.suptitle('K-Means Clustering — Behavioral + Artist Features (PCA projection)', fontweight='bold')
    plt.tight_layout()
    plt.savefig('kmeans_clustering.png', dpi=150)
    plt.show()
    print("\nPlot saved to kmeans_clustering.png")
