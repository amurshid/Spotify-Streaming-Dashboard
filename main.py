from data_loader import load_streaming_history, restrict_to_overlap
from dimensions import (hourly_vector, artist_vector, seasonal_vector,
                        dow_vector, skip_rate, track_diversity,
                        listen_duration_vector, engagement_vector)
from similarity import cosine_sim, scalar_sim, print_report
from classifier import (build_user_profile, build_track_features,
                        train_classifier, train_knn,
                        predict_liked, predict_random_song)
from clustering import run_kmeans_clustering

MY_FILES = [
    'my_spotify_data/StreamingHistory_music_0.json',
    'my_spotify_data/StreamingHistory_music_1.json',
]
ATHARVA_FILES = [
    'atharva_more_spotify_data/StreamingHistory_music_0.json',
    'atharva_more_spotify_data/StreamingHistory_music_1.json',
]


def main():
    df_me = load_streaming_history(MY_FILES)
    df_at = load_streaming_history(ATHARVA_FILES)
    df_me, df_at = restrict_to_overlap(df_me, df_at)
    print(f"Records — Me: {len(df_me):,}  |  Atharva: {len(df_at):,}\n")

    # ── 1. Similarity measure ────────────────────────────────────────────────
    print("=" * 45)
    print("1. SIMILARITY MEASURE")
    scores = {
        'artist':          cosine_sim(artist_vector(df_me),          artist_vector(df_at)),
        'hourly':          cosine_sim(hourly_vector(df_me),          hourly_vector(df_at)),
        'seasonality':     cosine_sim(seasonal_vector(df_me),        seasonal_vector(df_at)),
        'day_of_week':     cosine_sim(dow_vector(df_me),             dow_vector(df_at)),
        'listen_duration': cosine_sim(listen_duration_vector(df_me), listen_duration_vector(df_at)),
        'engagement':      cosine_sim(engagement_vector(df_me),      engagement_vector(df_at)),
        'skip_rate':       scalar_sim(skip_rate(df_me),              skip_rate(df_at)),
        'diversity':       scalar_sim(track_diversity(df_me),        track_diversity(df_at)),
    }
    final_sim = print_report(scores)

    # ── 2. Clustering ────────────────────────────────────────────────────────
    print("\n" + "=" * 45)
    print("2. K-MEANS CLUSTERING (k=2)")
    _, _, ari = run_kmeans_clustering(df_me, df_at)

    # ── 3. Decision Tree classifier (me) ─────────────────────────────────────
    print("\n" + "=" * 45)
    print("3. DECISION TREE — Would I like a song?")
    my_profile = build_user_profile(df_me)
    my_features = build_track_features(df_me, my_profile)
    clf, dt_acc = train_classifier(my_features)

    recs = predict_liked(clf, df_at, my_profile)
    print(f"\n  Songs from Atharva I'd likely enjoy ({len(recs)} total, showing top 10):")
    print(recs.head(10).to_string(index=False))

    print("\n  Random song prediction for me:")
    predict_random_song(clf, df_at, my_profile, listener_name='Me')

    # ── 4. KNN classifier (Atharva) ──────────────────────────────────────────
    print("\n" + "=" * 45)
    print("4. KNN (k=5) — Would Atharva like a song?")
    atharva_profile  = build_user_profile(df_at)
    atharva_features = build_track_features(df_at, atharva_profile)
    knn, knn_acc = train_knn(atharva_features)

    print("\n  Random song prediction for Atharva:")
    predict_random_song(knn, df_me, atharva_profile, listener_name='Atharva')

    # ── 5. Summary of findings ───────────────────────────────────────────────
    weakest_dim  = min(scores, key=scores.get)
    strongest_dim = max(scores, key=scores.get)
    separable = "YES" if ari > 0.3 else "PARTIALLY" if ari > 0.1 else "NO"

    print("\n" + "=" * 55)
    print("SUMMARY OF FINDINGS")
    print("=" * 55)
    print(f"  Overall similarity score  : {final_sim:.4f}")
    print(f"  Most similar dimension    : {strongest_dim} ({scores[strongest_dim]:.4f})")
    print(f"  Least similar dimension   : {weakest_dim} ({scores[weakest_dim]:.4f})")
    print()
    print(f"  Decision Tree accuracy    : {dt_acc:.1%}  (predicts if I'd like a song)")
    print(f"  KNN accuracy              : {knn_acc:.1%}  (predicts if Atharva would like a song)")
    print()
    print(f"  Clustering ARI            : {ari:.4f}  →  listeners separable? {separable}")
    print(f"  Songs recommended to me from Atharva's library: {len(recs)}")
    print("=" * 55)


if __name__ == '__main__':
    main()
