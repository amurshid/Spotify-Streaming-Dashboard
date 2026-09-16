"""
Property tests for the feature vectors.

Every vector the pipeline builds is a proportion over a fixed number of bins.
Two things must hold for all of them: the length is fixed regardless of input,
and the values sum to 1 unless there is nothing to normalise — in which case
they must be zeros rather than NaN. Each function guards division by zero, and
nothing exercised those guards before these tests.
"""
import numpy as np
import pytest

from conftest import make_plays, COMPLETE_MS, SKIP_MS
from dimensions import (
    hourly_vector, seasonal_vector, dow_vector, listen_duration_vector,
    engagement_vector, skip_rate, track_diversity, artist_vector,
    track_hourly_vector, track_seasonal_vector, track_dow_vector,
    track_listen_duration_vector, track_engagement_vector, track_skip_rate,
    _duration_bucket, DURATION_LABELS,
)

USER_VECTORS = [
    (hourly_vector, 24),
    (seasonal_vector, 4),
    (dow_vector, 7),
    (listen_duration_vector, 6),
    (engagement_vector, 3),
]

TRACK_VECTORS = [
    (track_hourly_vector, 24),
    (track_seasonal_vector, 4),
    (track_dow_vector, 7),
    (track_listen_duration_vector, 6),
    (track_engagement_vector, 3),
]


@pytest.mark.parametrize('fn,length', USER_VECTORS + TRACK_VECTORS)
def test_vector_has_fixed_length(fn, length, simple_plays):
    assert len(fn(simple_plays)) == length


@pytest.mark.parametrize('fn,length', USER_VECTORS + TRACK_VECTORS)
def test_vector_is_a_proportion(fn, length, simple_plays):
    assert fn(simple_plays).sum() == pytest.approx(1.0)


@pytest.mark.parametrize('fn,length', USER_VECTORS + TRACK_VECTORS)
def test_empty_input_gives_zeros_not_nan(fn, length, empty_plays):
    """The zero-division guard in every vector function."""
    vec = fn(empty_plays)
    assert len(vec) == length
    assert not np.isnan(vec).any()
    assert vec.sum() == 0.0


def test_hourly_vector_places_mass_on_the_right_hour(simple_plays):
    vec = hourly_vector(simple_plays)
    assert vec[9] == pytest.approx(1.0)
    assert vec[np.arange(24) != 9].sum() == pytest.approx(0.0)


def test_skip_rate_counts_plays_under_thirty_seconds(simple_plays):
    assert skip_rate(simple_plays) == pytest.approx(0.25)


def test_skip_rate_of_empty_history_is_zero(empty_plays):
    assert skip_rate(empty_plays) == 0.0


def test_track_diversity_is_unique_tracks_over_plays(simple_plays):
    assert track_diversity(simple_plays) == pytest.approx(3 / 4)


def test_artist_vector_is_a_proportion_of_minutes(simple_plays):
    vec = artist_vector(simple_plays)
    assert set(vec) == {'Artist X', 'Artist Y'}
    assert sum(vec.values()) == pytest.approx(1.0)


def test_artist_vector_of_empty_history_is_empty(empty_plays):
    assert artist_vector(empty_plays) == {}


@pytest.mark.parametrize('ms,expected', [
    (0,       '<10s'),
    (9_999,   '<10s'),
    (10_000,  '10-30s'),
    (29_999,  '10-30s'),
    (30_000,  '30-60s'),
    (120_000, '2-3min'),
    (180_000, '3min+'),
    (10 ** 9, '3min+'),
])
def test_duration_bucket_boundaries(ms, expected):
    """Boundary values land in the higher bucket, not the lower one."""
    assert _duration_bucket(ms) == expected
    assert expected in DURATION_LABELS


def test_engagement_vector_splits_skipped_partial_complete():
    rows = [
        ('T', 'A', '2024-01-01 10:00', SKIP_MS),        # skipped
        ('T', 'A', '2024-01-01 11:00', 60_000),         # partial
        ('T', 'A', '2024-01-01 12:00', COMPLETE_MS),    # complete
        ('T', 'A', '2024-01-01 13:00', COMPLETE_MS),    # complete
    ]
    vec = engagement_vector(make_plays(rows))
    assert vec == pytest.approx([0.25, 0.25, 0.5])


def test_track_skip_rate_matches_the_group(simple_plays):
    group = simple_plays[simple_plays['trackName'] == 'Track B']
    assert track_skip_rate(group) == pytest.approx(1.0)
