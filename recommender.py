import pandas as pd
from data import p1, p2
from classifier import (
    build_user_profile, build_track_features,
    train_classifier, train_knn, FEATURES, _similarity_features,
)


def _adapt(df):
    d = df.copy()
    if 'minutesPlayed' not in d.columns:
        d['minutesPlayed'] = d['minPlayed']
    return d


def _get_recs(clf, profile, candidate_df):
    rows = []
    for (track, artist), group in candidate_df.groupby(['trackName', 'artistName']):
        play_count = len(group)
        feats = _similarity_features(group, artist, profile)
        feats.update({
            'track':      track,
            'artist':     artist,
            'play_count': play_count,
            'skip_count': int((group['msPlayed'] < 30_000).sum()),
            'is_repeat':  int(play_count > 1),
        })
        rows.append(feats)

    cands = pd.DataFrame(rows)
    X = cands[FEATURES]
    cands['liked']      = clf.predict(X)
    cands['confidence'] = clf.predict_proba(X)[:, 1]

    liked = cands[cands['liked'] == 1].sort_values('confidence', ascending=False)
    return liked[['track', 'artist', 'confidence', 'play_count']].reset_index(drop=True)


def _build():
    p1a, p2a = _adapt(p1), _adapt(p2)

    p1_profile = build_user_profile(p1a)
    p2_profile = build_user_profile(p2a)
    p1_feats   = build_track_features(p1a, p1_profile)
    p2_feats   = build_track_features(p2a, p2_profile)

    p1_dt,  p1_dt_acc,  p1_dt_cm  = train_classifier(p1_feats)
    p1_knn, p1_knn_acc, p1_knn_cm = train_knn(p1_feats)
    p2_dt,  p2_dt_acc,  p2_dt_cm  = train_classifier(p2_feats)
    p2_knn, p2_knn_acc, p2_knn_cm = train_knn(p2_feats)

    return {
        ('p1', 'dt'):  (_get_recs(p1_dt,  p1_profile, p2a), p1_dt_acc,  p1_dt_cm),
        ('p1', 'knn'): (_get_recs(p1_knn, p1_profile, p2a), p1_knn_acc, p1_knn_cm),
        ('p2', 'dt'):  (_get_recs(p2_dt,  p2_profile, p1a), p2_dt_acc,  p2_dt_cm),
        ('p2', 'knn'): (_get_recs(p2_knn, p2_profile, p1a), p2_knn_acc, p2_knn_cm),
    }


print("Training recommendation models…")
RECS = _build()
