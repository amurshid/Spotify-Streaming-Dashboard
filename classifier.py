import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from dimensions import (artist_vector, hourly_vector, seasonal_vector, dow_vector,
                        skip_rate, listen_duration_vector, engagement_vector,
                        track_hourly_vector, track_seasonal_vector, track_dow_vector,
                        track_skip_rate, track_listen_duration_vector, track_engagement_vector)
from similarity import cosine_sim, scalar_sim


def build_user_profile(df):
    """Compute user-level profile vectors used as the reference for similarity features."""
    return {
        'artist':          artist_vector(df),
        'hourly':          hourly_vector(df),
        'seasonal':        seasonal_vector(df),
        'dow':             dow_vector(df),
        'skip_rate':       skip_rate(df),
        'listen_duration': listen_duration_vector(df),
        'engagement':      engagement_vector(df),
    }


def _similarity_features(group, artist, profile):
    """
    Compute per-track similarity scores against a user profile.
    Each score is in [0, 1] — higher means the track's pattern matches the user better.
    """
    return {
        'artist_sim':   cosine_sim({artist: 1.0},                    profile['artist']),
        'hourly_sim':   cosine_sim(track_hourly_vector(group),        profile['hourly']),
        'seasonal_sim': cosine_sim(track_seasonal_vector(group),      profile['seasonal']),
        'dow_sim':      cosine_sim(track_dow_vector(group),           profile['dow']),
        'duration_sim': cosine_sim(track_listen_duration_vector(group), profile['listen_duration']),
        'engage_sim':   cosine_sim(track_engagement_vector(group),    profile['engagement']),
    }


FEATURES = [
    'skip_count',
    'artist_sim', 'hourly_sim', 'seasonal_sim', 'dow_sim',
    'duration_sim', 'engage_sim',
]


def build_track_features(df, user_profile=None):
    """
    Build one row per unique (trackName, artistName).
    Features are similarity scores between each track's listening patterns and the user profile.
    Liked = played more than 2 times AND never skipped (all plays > 30s).
    """
    if user_profile is None:
        user_profile = build_user_profile(df)

    rows = []
    for (track, artist), group in df.groupby(['trackName', 'artistName']):
        play_count = len(group)
        skip_count = (group['msPlayed'] < 30000).sum()
        liked = int(play_count > 1)

        feats = _similarity_features(group, artist, user_profile)
        feats.update({
            'track': track, 'artist': artist,
            'play_count': play_count, 'skip_count': int(skip_count),
            'is_repeat': int(play_count > 1), 'liked': liked,
        })
        rows.append(feats)

    return pd.DataFrame(rows)


def train_classifier(df_features, max_depth=5):
    X = df_features[FEATURES]
    y = df_features['liked']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = (y_pred == y_test).mean()

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    top_feature = FEATURES[clf.feature_importances_.argmax()]

    print(f"  Tracks: {len(df_features):,}  |  Liked: {y.sum()}  |  Not liked: {(y==0).sum()}")
    print(f"  Accuracy : {acc:.1%}")
    print(f"  Confusion Matrix:")
    print(f"              Predicted")
    print(f"              Not Liked  Liked")
    print(f"  Actual Not Liked  {tn:>6}   {fp:>5}")
    print(f"  Actual Liked      {fn:>6}   {tp:>5}")
    print(f"  Top split feature: {top_feature}")

    return clf, acc, cm


def train_knn(df_features, n_neighbors=5):
    X = df_features[FEATURES]
    y = df_features['liked']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    acc = (y_pred == y_test).mean()

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"  Tracks: {len(df_features):,}  |  Liked: {y.sum()}  |  Not liked: {(y==0).sum()}")
    print(f"  Accuracy : {acc:.1%}")
    print(f"  Confusion Matrix:")
    print(f"              Predicted")
    print(f"              Not Liked  Liked")
    print(f"  Actual Not Liked  {tn:>6}   {fp:>5}")
    print(f"  Actual Liked      {fn:>6}   {tp:>5}")

    return knn, acc, cm


def predict_liked(clf, candidate_df, listener_profile):
    """
    Given candidate tracks and a listener's profile, predict which the listener would like.
    Similarity features are computed relative to the listener's profile, not the candidate's.
    """
    rows = []
    for (track, artist), group in candidate_df.groupby(['trackName', 'artistName']):
        play_count = len(group)
        feats = _similarity_features(group, artist, listener_profile)
        feats.update({
            'track': track, 'artist': artist,
            'play_count': play_count,
            'skip_count': int((group['msPlayed'] < 30_000).sum()),
            'is_repeat': int(play_count > 1),
        })
        rows.append(feats)

    candidates = pd.DataFrame(rows)
    candidates['predicted_liked'] = clf.predict(candidates[FEATURES])
    return candidates[candidates['predicted_liked'] == 1][['track', 'artist']].reset_index(drop=True)


def predict_random_song(clf, picker_df, listener_profile, listener_name='listener'):
    """
    Randomly pick one song from picker_df, predict if listener would like it.
    picker_df:        the user whose library is being sampled (e.g. Atharva)
    listener_profile: precomputed profile of the user being judged (e.g. me)
    """
    unique_tracks = picker_df[['trackName', 'artistName']].drop_duplicates()
    song = unique_tracks.sample(1).iloc[0]
    track, artist = song['trackName'], song['artistName']

    group = picker_df[(picker_df['trackName'] == track) & (picker_df['artistName'] == artist)]
    play_count = len(group)
    feats = _similarity_features(group, artist, listener_profile)
    feats['play_count'] = play_count
    feats['skip_count'] = int((group['msPlayed'] < 30_000).sum())
    feats['is_repeat']  = int(play_count > 1)

    X = pd.DataFrame([feats])[FEATURES]
    prediction = clf.predict(X)[0]
    label = 'LIKED' if prediction == 1 else 'NOT LIKED'

    print(f'  "{track}" by {artist}  →  {label}')

    return track, artist, prediction
