import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                             average_precision_score, precision_score, recall_score)

from dimensions import (artist_vector, hourly_vector, seasonal_vector, dow_vector,
                        listen_duration_vector, engagement_vector,
                        track_hourly_vector, track_seasonal_vector, track_dow_vector,
                        track_skip_rate, track_listen_duration_vector,
                        track_engagement_vector)
from similarity import cosine_sim

# Fraction of the listening history used to build features. The remainder is
# held back to derive the label.
SPLIT_FRAC = 0.8

# Features are computed strictly from the history window, so none of them can
# see the outcome they are used to predict.
FEATURES = [
    'play_count', 'skip_rate',
    'artist_sim', 'hourly_sim', 'seasonal_sim', 'dow_sim',
    'duration_sim', 'engage_sim',
]

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def temporal_split(df, frac=SPLIT_FRAC):
    """
    Split a listening history chronologically.

    Features are built from the earlier window and the label from the later one,
    so a track's outcome is never derived from the same plays that describe it.
    A random row split would not give this: every feature is aggregated over a
    track's plays, so a label derived from those same plays leaks through the
    group size into every feature at once.
    """
    cutoff  = df['endTime'].quantile(frac)
    history = df[df['endTime'] <  cutoff].copy()
    future  = df[df['endTime'] >= cutoff].copy()
    return history, future, cutoff


def build_user_profile(df):
    """Compute user-level profile vectors used as the reference for similarity features."""
    return {
        'artist':          artist_vector(df),
        'hourly':          hourly_vector(df),
        'seasonal':        seasonal_vector(df),
        'dow':             dow_vector(df),
        'listen_duration': listen_duration_vector(df),
        'engagement':      engagement_vector(df),
    }


def _similarity_features(group, artist, profile):
    """
    Compute per-track similarity scores against a user profile.
    Each score is in [0, 1] — higher means the track's pattern matches the user better.
    """
    return {
        'artist_sim':   cosine_sim({artist: 1.0},                      profile['artist']),
        'hourly_sim':   cosine_sim(track_hourly_vector(group),          profile['hourly']),
        'seasonal_sim': cosine_sim(track_seasonal_vector(group),        profile['seasonal']),
        'dow_sim':      cosine_sim(track_dow_vector(group),             profile['dow']),
        'duration_sim': cosine_sim(track_listen_duration_vector(group), profile['listen_duration']),
        'engage_sim':   cosine_sim(track_engagement_vector(group),      profile['engagement']),
    }


def _track_rows(df, profile):
    """One row of features per unique (trackName, artistName) in df."""
    rows = []
    for (track, artist), group in df.groupby(['trackName', 'artistName']):
        feats = _similarity_features(group, artist, profile)
        feats.update({
            'track':      track,
            'artist':     artist,
            'play_count': len(group),
            'skip_rate':  track_skip_rate(group),
        })
        rows.append(feats)
    return pd.DataFrame(rows)


def build_track_features(df, user_profile=None, frac=SPLIT_FRAC):
    """
    Build one training row per unique (trackName, artistName).

    Features describe how a track was listened to during the history window.
    Label — returned = the listener came back to the track during the held-out
    future window. Spotify's export carries no explicit rating, so a later
    return to a track is the available proxy for preference.
    """
    history, future, _ = temporal_split(df, frac)

    if user_profile is None:
        user_profile = build_user_profile(history)

    features = _track_rows(history, user_profile)
    if features.empty:
        return features

    returned = set(zip(future['trackName'], future['artistName']))
    features['returned'] = [
        int((t, a) in returned) for t, a in zip(features['track'], features['artist'])
    ]
    return features


def _evaluate(model, X, y, name):
    """Fit on a stratified split, score against a majority-class baseline."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)

    cv_scores = cross_val_score(model, X, y, cv=CV, scoring='roc_auc')
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    metrics = {
        'model':          name,
        'accuracy':       float((y_pred == y_test).mean()),
        'baseline_acc':   float(dummy.score(X_test, y_test)),
        'precision':      float(precision_score(y_test, y_pred, zero_division=0)),
        'recall':         float(recall_score(y_test, y_pred, zero_division=0)),
        'roc_auc':        float(roc_auc_score(y_test, y_score)),
        'pr_auc':         float(average_precision_score(y_test, y_score)),
        'baseline_pr':    float(y_test.mean()),
        'cv_auc_mean':    float(cv_scores.mean()),
        'cv_auc_std':     float(cv_scores.std()),
        'cm':             cm,
        'positive_rate':  float(y.mean()),
        'n_samples':      int(len(y)),
    }
    return model, metrics


def print_metrics(m):
    tn, fp, fn, tp = m['cm'].ravel()
    lift = m['accuracy'] - m['baseline_acc']

    print(f"  {m['model']}")
    print(f"    Tracks            : {m['n_samples']:,}   (returned: {m['positive_rate']:.1%})")
    print(f"    Accuracy          : {m['accuracy']:.1%}   baseline {m['baseline_acc']:.1%}"
          f"   ({lift:+.1%})")
    print(f"    ROC-AUC           : {m['roc_auc']:.3f}   (0.500 = no better than chance)")
    print(f"    PR-AUC            : {m['pr_auc']:.3f}   baseline {m['baseline_pr']:.3f}")
    print(f"    5-fold CV ROC-AUC : {m['cv_auc_mean']:.3f} +/- {m['cv_auc_std']:.3f}")
    print(f"    Precision / Recall: {m['precision']:.1%} / {m['recall']:.1%}")
    print(f"    Confusion matrix  : TN {tn}  FP {fp}  FN {fn}  TP {tp}")


def train_classifier(df_features, max_depth=5):
    X, y = df_features[FEATURES], df_features['returned']
    clf = DecisionTreeClassifier(max_depth=max_depth, class_weight='balanced',
                                 random_state=42)
    clf, metrics = _evaluate(clf, X, y, f'Decision Tree (max_depth={max_depth})')

    importances = clf.feature_importances_
    metrics['top_feature'] = FEATURES[int(importances.argmax())]
    metrics['importances'] = dict(zip(FEATURES, importances.round(4)))

    print_metrics(metrics)
    print(f"    Top split feature : {metrics['top_feature']}")
    return clf, metrics


def train_knn(df_features, n_neighbors=5):
    X, y = df_features[FEATURES], df_features['returned']

    # KNN is distance-based and play_count is an unbounded count while every
    # similarity feature is bounded to [0, 1]. Without scaling, play_count alone
    # would dictate every neighbourhood. The scaler sits inside the pipeline so
    # it is refit on each CV fold rather than on the full dataset.
    knn = Pipeline([
        ('scaler', StandardScaler()),
        ('knn',    KNeighborsClassifier(n_neighbors=n_neighbors)),
    ])
    knn, metrics = _evaluate(knn, X, y, f'KNN (k={n_neighbors})')

    print_metrics(metrics)
    return knn, metrics


def predict_liked(clf, candidate_df, listener_profile):
    """
    Score another listener's library against this listener's profile and return
    the tracks the model predicts they would return to.
    """
    candidates = _track_rows(candidate_df, listener_profile)
    candidates['predicted'] = clf.predict(candidates[FEATURES])
    return candidates[candidates['predicted'] == 1][['track', 'artist']].reset_index(drop=True)


def predict_random_song(clf, picker_df, listener_profile, listener_name='listener'):
    """
    Sample one track from picker_df and predict whether the listener behind
    listener_profile would return to it.
    """
    unique = picker_df[['trackName', 'artistName']].drop_duplicates()
    song = unique.sample(1, random_state=None).iloc[0]
    track, artist = song['trackName'], song['artistName']

    group = picker_df[(picker_df['trackName'] == track) &
                      (picker_df['artistName'] == artist)]

    feats = _similarity_features(group, artist, listener_profile)
    feats['play_count'] = len(group)
    feats['skip_rate']  = track_skip_rate(group)

    X = pd.DataFrame([feats])[FEATURES]
    prediction = int(clf.predict(X)[0])
    confidence = float(clf.predict_proba(X)[0, 1])

    label = 'WOULD RETURN' if prediction else 'WOULD NOT RETURN'
    print(f'  "{track}" by {artist}  ->  {label}  ({confidence:.0%} confidence)')

    return track, artist, prediction
