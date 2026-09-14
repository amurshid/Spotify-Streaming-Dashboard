import pandas as pd
from data import p1, p2
from classifier import (temporal_split, build_user_profile, build_track_features,
                        train_classifier, train_knn, FEATURES, _track_rows)


def _adapt(df):
    d = df.copy()
    if 'minutesPlayed' not in d.columns:
        d['minutesPlayed'] = d['minPlayed']
    return d


def _fit(df):
    """Profile and features come from the history window; the label from the future."""
    history, _, _ = temporal_split(df)
    profile = build_user_profile(history)
    return profile, build_track_features(df, profile)


def _get_recs(model, profile, candidate_df):
    cands = _track_rows(candidate_df, profile)
    X = cands[FEATURES]
    cands['predicted']  = model.predict(X)
    cands['confidence'] = model.predict_proba(X)[:, 1]

    liked = cands[cands['predicted'] == 1].sort_values('confidence', ascending=False)
    return liked[['track', 'artist', 'confidence', 'play_count']].reset_index(drop=True)


def _build():
    p1a, p2a = _adapt(p1), _adapt(p2)

    p1_profile, p1_feats = _fit(p1a)
    p2_profile, p2_feats = _fit(p2a)

    print("  Person 1:")
    p1_dt,  p1_dt_m  = train_classifier(p1_feats)
    p1_knn, p1_knn_m = train_knn(p1_feats)
    print("  Person 2:")
    p2_dt,  p2_dt_m  = train_classifier(p2_feats)
    p2_knn, p2_knn_m = train_knn(p2_feats)

    return {
        ('p1', 'dt'):  (_get_recs(p1_dt,  p1_profile, p2a), p1_dt_m),
        ('p1', 'knn'): (_get_recs(p1_knn, p1_profile, p2a), p1_knn_m),
        ('p2', 'dt'):  (_get_recs(p2_dt,  p2_profile, p1a), p2_dt_m),
        ('p2', 'knn'): (_get_recs(p2_knn, p2_profile, p1a), p2_knn_m),
    }


print("Training recommendation models…")
RECS = _build()
