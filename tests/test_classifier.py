"""
Contract tests for the preference classifier.

The tests that matter here are the leakage ones. The original label was
computed from the same plays as the features, so it bled into all of them at
once through the group size — and no train/test split could separate them,
because the leak was in the feature construction. The temporal split fixed it.
These tests pin that fix down, so a future change that reintroduces it fails
here rather than silently inflating a score.
"""
import pytest

from conftest import make_plays, COMPLETE_MS, SKIP_MS
from classifier import (
    FEATURES, SPLIT_FRAC, temporal_split, build_user_profile,
    build_track_features, train_classifier, train_knn, _track_rows,
)


def split_frame(history_plays, future_plays, n_filler=40):
    """
    Build a frame whose split point is known.

    History plays land in January, filler runs through February, future plays
    land in April. The filler dominates, so the 80% cutoff always falls inside
    it and never among the plays a test is reasoning about. Tests assert the
    cutoff landed where intended rather than trusting this.
    """
    rows = [('Track A', 'Artist X', f'2024-01-{d:02d} 10:00', ms)
            for d, ms in history_plays]
    rows += [(f'Filler {i}', f'Artist {i % 7}', f'2024-02-{(i % 28) + 1:02d} 12:00',
              COMPLETE_MS) for i in range(n_filler)]
    rows += [('Track A', 'Artist X', f'2024-04-{d:02d} 10:00', ms)
             for d, ms in future_plays]
    return make_plays(rows)


# ── The split itself ─────────────────────────────────────────────────────────

def test_split_windows_are_disjoint_and_ordered(training_plays):
    history, future, cutoff = temporal_split(training_plays)
    assert len(history) + len(future) == len(training_plays)
    assert history['endTime'].max() < cutoff <= future['endTime'].min()


def test_split_respects_the_requested_fraction(training_plays):
    history, _, _ = temporal_split(training_plays, frac=SPLIT_FRAC)
    assert len(history) / len(training_plays) == pytest.approx(SPLIT_FRAC, abs=0.05)


# ── Leakage ──────────────────────────────────────────────────────────────────

def test_features_are_built_only_from_the_history_window():
    """
    A track played twice before the split and five more times after must report
    a play count of two. If the future window reaches the features at all, this
    is where it shows.
    """
    df = split_frame(history_plays=[(1, COMPLETE_MS), (2, COMPLETE_MS)],
                     future_plays=[(d, COMPLETE_MS) for d in range(1, 6)])

    _, _, cutoff = temporal_split(df)
    assert cutoff.month == 2, 'split landed outside the filler band'

    feats = build_track_features(df)
    track_a = feats[feats['track'] == 'Track A'].iloc[0]

    assert track_a['play_count'] == 2
    assert track_a['returned'] == 1


def test_skip_rate_ignores_the_future_window():
    """Every history play completes and every future play is a skip."""
    df = split_frame(history_plays=[(d, COMPLETE_MS) for d in range(1, 4)],
                     future_plays=[(d, SKIP_MS) for d in range(1, 8)])

    _, _, cutoff = temporal_split(df)
    assert cutoff.month == 2, 'split landed outside the filler band'

    feats = build_track_features(df)
    track_a = feats[feats['track'] == 'Track A'].iloc[0]

    assert track_a['skip_rate'] == 0.0


def test_a_track_only_in_the_future_window_has_no_row():
    """Nothing is known about it at prediction time, so it cannot be scored."""
    rows = [('Old', f'Artist {i}', f'2024-01-{i:02d} 10:00', COMPLETE_MS)
            for i in range(1, 20)]
    rows += [('Brand New', 'Artist Z', '2024-03-01 10:00', COMPLETE_MS)]

    feats = build_track_features(make_plays(rows))
    assert 'Brand New' not in set(feats['track'])


def test_label_is_not_among_the_features():
    assert 'returned' not in FEATURES


def test_no_feature_is_derived_from_the_future(training_plays):
    """
    Features built by the public entry point must match those built directly
    from the history window. Any feature reading past the cutoff breaks this.
    """
    history, _, _ = temporal_split(training_plays)
    profile = build_user_profile(history)

    produced = build_track_features(training_plays, profile)
    expected = _track_rows(history, profile)

    merged = produced.merge(expected, on=['track', 'artist'], suffixes=('_out', '_hist'))
    assert len(merged) == len(produced) == len(expected)
    for feature in FEATURES:
        assert merged[f'{feature}_out'].equals(merged[f'{feature}_hist']), feature


# ── Feature frame contract ───────────────────────────────────────────────────

def test_every_declared_feature_is_produced(training_plays):
    feats = build_track_features(training_plays)
    assert set(FEATURES).issubset(feats.columns)
    assert 'returned' in feats.columns


def test_similarity_features_are_bounded(training_plays):
    feats = build_track_features(training_plays)
    for feature in [f for f in FEATURES if f.endswith('_sim')]:
        assert feats[feature].between(0.0, 1.0).all(), feature


def test_features_have_no_missing_values(training_plays):
    feats = build_track_features(training_plays)
    assert not feats[FEATURES].isna().any().any()


# ── Training ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('train', [train_classifier, train_knn])
def test_training_reports_a_full_metric_set(train, training_plays):
    _, metrics = train(build_track_features(training_plays))

    for key in ['accuracy', 'baseline_acc', 'roc_auc', 'pr_auc',
                'cv_auc_mean', 'cv_auc_std', 'precision', 'recall', 'cm']:
        assert key in metrics, key

    assert 0.0 <= metrics['roc_auc'] <= 1.0
    assert 0.0 <= metrics['accuracy'] <= 1.0


@pytest.mark.parametrize('train', [train_classifier, train_knn])
def test_confusion_matrix_is_always_two_by_two(train, training_plays):
    """
    It is built with explicit labels. Without them a split containing a single
    class returns a 1x1 matrix and unpacking it raises.
    """
    _, metrics = train(build_track_features(training_plays))
    assert metrics['cm'].shape == (2, 2)
    assert len(metrics['cm'].ravel()) == 4


def test_knn_scales_its_features(training_plays):
    """
    play_count is an unbounded count while every similarity feature is bounded
    to [0, 1], so without a scaler the count alone decides every neighbourhood.
    The scaler belongs inside the pipeline so it is refit per CV fold.
    """
    model, _ = train_knn(build_track_features(training_plays))
    assert 'scaler' in model.named_steps
