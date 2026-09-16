from data import p1, p2
from dimensions import (
    hourly_vector, artist_vector, seasonal_vector, dow_vector,
    skip_rate, track_diversity, listen_duration_vector, engagement_vector,
)
from similarity import cosine_sim, scalar_sim, compute_final_score, WEIGHTS  # noqa: F401 (re-exported)
from clustering import cluster_tracks


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


print("Computing similarity & clustering analytics…")
SIM_SCORES, SIM_FINAL = _run_similarity()
CLUSTER_DATA = cluster_tracks(_adapt(p1), _adapt(p2), use_artist=True)
CLUSTER_BEHAVIOUR = cluster_tracks(_adapt(p1), _adapt(p2), use_artist=False)
