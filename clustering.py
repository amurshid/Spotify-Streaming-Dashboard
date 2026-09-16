import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score

from dimensions import (
    track_hourly_vector, track_seasonal_vector, track_dow_vector,
    track_listen_duration_vector, track_engagement_vector, track_skip_rate,
)

# Behavioural block: 24 hourly + 4 seasonal + 7 day-of-week + 6 duration
# + 3 engagement + skip rate + play count.
N_BEHAVIOURAL = 46
N_ARTIST_COMPONENTS = 50


def build_artist_vocab(df_me, df_at):
    """Sorted dict of all unique artists across both users → index."""
    artists = set(df_me['artistName'].unique()) | set(df_at['artistName'].unique())
    return {a: i for i, a in enumerate(sorted(artists))}


def build_track_raw_features(df, user_label, artist_vocab):
    """
    One row per unique (trackName, artistName), holding the behavioural block
    and the artist one-hot separately so each can be reduced on its own terms.
    """
    rows = []
    for (track, artist), group in df.groupby(['trackName', 'artistName']):
        behavioural = np.concatenate([
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
            'behavioural': behavioural,
            'artist_onehot': artist_vec,
        })
    return rows


def _reduce(all_rows, use_artist, random_state):
    """
    Reduce each block on its own terms, then concatenate.

    The two blocks cannot share a preprocessing path. Standardising a sparse
    one-hot rescales every near-empty artist column up to unit variance, which
    spreads the variance evenly across all columns and leaves PCA nothing
    concentrated to find. TruncatedSVD works on the raw indicator matrix
    without centring it, which is what keeps the artist structure intact.
    """
    behavioural = np.vstack([r['behavioural'] for r in all_rows])
    blocks = [StandardScaler().fit_transform(behavioural)]
    explained = None

    if use_artist:
        artist = np.vstack([r['artist_onehot'] for r in all_rows])
        n_components = min(N_ARTIST_COMPONENTS, artist.shape[1] - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        reduced = svd.fit_transform(artist)
        explained = svd.explained_variance_ratio_.sum()

        # Put the artist block on the same footing as the behavioural one, so
        # neither dominates the distance purely through its own scale.
        blocks.append(StandardScaler().fit_transform(reduced))

    return np.hstack(blocks), explained


def cluster_tracks(df_me, df_at, use_artist=True, n_clusters=2, random_state=42):
    """Cluster both users' tracks and score the split against true identity."""
    artist_vocab = build_artist_vocab(df_me, df_at)
    all_rows = (build_track_raw_features(df_me, 0, artist_vocab) +
                build_track_raw_features(df_at, 1, artist_vocab))

    X, explained = _reduce(all_rows, use_artist, random_state)
    true_labels = np.array([r['user'] for r in all_rows])

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = km.fit_predict(X)

    pca = PCA(n_components=2, random_state=random_state)
    X_2d = pca.fit_transform(X)

    distances = km.transform(X).min(axis=1)
    outliers = pd.DataFrame({
        'track':    [r['track']  for r in all_rows],
        'artist':   [r['artist'] for r in all_rows],
        'user':     ['Person 1' if r['user'] == 0 else 'Person 2' for r in all_rows],
        'distance': distances,
        'cluster':  cluster_labels,
    }).sort_values('distance', ascending=False).reset_index(drop=True)

    cluster_stats = []
    for c in range(n_clusters):
        mask = cluster_labels == c
        total = int(mask.sum())
        p1 = int(((true_labels == 0) & mask).sum())
        p2 = int(((true_labels == 1) & mask).sum())
        cluster_stats.append({
            'cluster': c, 'total': total, 'person1': p1, 'person2': p2,
            'dominant': 'Person 1' if p1 >= p2 else 'Person 2',
            'purity': round(max(p1, p2) / total * 100, 1) if total else 0.0,
        })

    return {
        'X_2d': X_2d, 'var': pca.explained_variance_ratio_,
        'cluster_labels': cluster_labels, 'true_labels': true_labels,
        'ari': adjusted_rand_score(true_labels, cluster_labels),
        'sil': silhouette_score(X, cluster_labels),
        'artist_variance': explained,
        'n_artists': len(artist_vocab),
        'n_tracks': len(all_rows),
        'outliers': outliers, 'cluster_stats': cluster_stats,
    }


def run_kmeans_clustering(df_me, df_at, n_clusters=2, random_state=42):
    """
    Cluster twice — once on listening behaviour alone, once with artist
    identity added — and report both.

    Adding artists does not help, and the reason is in the encoding rather than
    the data. A track's one-hot has a single 1, so any two tracks by different
    artists are orthogonal whether or not they belong to the same listener. The
    block therefore carries artist identity but nothing about which listener an
    artist belongs to, and K-Means has no co-occurrence structure to exploit.
    Recovering that would need an artist representation learned from
    co-listening, which cannot be built here without using the labels.
    """
    print("\n--- K-Means Clustering (k=2) ---\n")

    behaviour = cluster_tracks(df_me, df_at, use_artist=False,
                               n_clusters=n_clusters, random_state=random_state)
    combined  = cluster_tracks(df_me, df_at, use_artist=True,
                               n_clusters=n_clusters, random_state=random_state)

    print(f"Tracks              : {combined['n_tracks']:,}")
    print(f"Artist vocabulary   : {combined['n_artists']:,} unique artists")
    print(f"Artist SVD variance : {combined['artist_variance']:.1%} retained "
          f"in {N_ARTIST_COMPONENTS} components\n")

    print(f"{'Feature set':<28} {'ARI':>8} {'Silhouette':>12}")
    print("-" * 50)
    for label, res in [('behaviour only', behaviour), ('behaviour + artist', combined)]:
        print(f"{label:<28} {res['ari']:>8.4f} {res['sil']:>12.4f}")
    print("-" * 50)
    print("ARI: 1.0 = perfect separation, 0.0 = random\n")

    _print_cluster_purity(combined['cluster_stats'])
    _print_outliers(combined['outliers'], top_n=10)
    _plot_pca(combined, behaviour)

    return combined['cluster_labels'], combined['true_labels'], combined['ari']


def _print_cluster_purity(cluster_stats):
    print("Cluster composition (behaviour + artist):")
    for s in cluster_stats:
        print(f"  Cluster {s['cluster']}: {s['total']:5d} tracks  |  "
              f"Person 1: {s['person1']:5d}  Person 2: {s['person2']:5d}"
              f"  → mostly {s['dominant']} ({s['purity']:.1f}% pure)")


def _print_outliers(df, top_n=10):
    def safe(s):
        return s.encode('ascii', 'replace').decode('ascii')

    print(f"\n--- Outlier detection: {top_n} most unusual tracks ---\n")
    print(f"{'Rank':<5} {'User':<10} {'Distance':>9}  Track - Artist")
    print("-" * 70)
    for rank, (_, row) in enumerate(df.head(top_n).iterrows(), start=1):
        print(f"{rank:<5} {row['user']:<10} {row['distance']:>9.4f}  "
              f"{safe(row['track'])} - {safe(row['artist'])}")


def _plot_pca(combined, behaviour):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    X_2d, var = combined['X_2d'], combined['var']

    for c, marker in [(0, 'o'), (1, 's')]:
        mask = combined['cluster_labels'] == c
        axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1], marker=marker,
                        alpha=0.45, s=18, label=f'Cluster {c}')
    axes[0].set_title(f"K-Means clusters (ARI = {combined['ari']:.3f})")
    axes[0].legend()

    for u, color, name in [(0, '#1f77b4', 'Person 1'), (1, '#ff7f0e', 'Person 2')]:
        mask = combined['true_labels'] == u
        axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1], c=color,
                        alpha=0.45, s=18, label=name)
    axes[1].set_title('True listener identity')
    axes[1].legend()

    for ax in axes[:2]:
        ax.set_xlabel(f'PC1 ({var[0]:.1%} var)')
        ax.set_ylabel(f'PC2 ({var[1]:.1%} var)')

    aris = [behaviour['ari'], combined['ari']]
    bars = axes[2].bar(['behaviour\nonly', 'behaviour\n+ artist'], aris,
                       color=['#888888', '#1DB954'])
    axes[2].axhline(0, color='#333', linewidth=0.8)
    axes[2].set_title('Separability by feature set')
    axes[2].set_ylabel('Adjusted Rand Index')
    # Both scores sit near zero, so a 0-1 axis would hide a negative bar
    # entirely. Keep 1.0 in view for scale but let the axis reach below zero.
    axes[2].set_ylim(min(-0.08, min(aris) * 1.5), 1.0)
    for bar, ari in zip(bars, aris):
        axes[2].text(bar.get_x() + bar.get_width() / 2,
                     ari + (0.03 if ari >= 0 else -0.06),
                     f'{ari:.4f}', ha='center', fontsize=10)
    axes[2].text(0.5, 0.55, '1.0 = perfect separation\n0.0 = random',
                 transform=axes[2].transAxes, ha='center',
                 fontsize=9, color='#666')

    plt.suptitle('K-Means Clustering — Behavioural and Artist Features',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig('kmeans_clustering.png', dpi=150)
    plt.close(fig)
    print("\nPlot saved to kmeans_clustering.png")
