"""
Tests for the clustering feature blocks.

The behavioural and artist blocks are reduced separately. These pin the block
shapes and the one-hot encoding, which is where the separability result comes
from — a track's indicator has a single 1, so tracks by different artists are
orthogonal whether or not they belong to the same listener.
"""
import numpy as np
import pytest

from conftest import make_plays, COMPLETE_MS
from clustering import (
    build_artist_vocab, build_track_raw_features, cluster_tracks, N_BEHAVIOURAL,
)


@pytest.fixture
def two_listeners():
    def frame(prefix, artists, hour):
        rows = []
        for i in range(40):
            rows.append((f'{prefix} Track {i}', artists[i % len(artists)],
                         f'2024-01-{(i % 28) + 1:02d} {hour:02d}:00', COMPLETE_MS))
        return make_plays(rows)

    return (frame('A', ['Artist P', 'Artist Q', 'Artist R'], 23),
            frame('B', ['Artist X', 'Artist Y', 'Artist Z'], 9))


def test_artist_vocab_covers_both_listeners(two_listeners):
    me, them = two_listeners
    vocab = build_artist_vocab(me, them)
    assert set(vocab) == {'Artist P', 'Artist Q', 'Artist R',
                          'Artist X', 'Artist Y', 'Artist Z'}
    assert sorted(vocab.values()) == list(range(len(vocab)))


def test_behavioural_block_has_the_declared_width(two_listeners):
    me, them = two_listeners
    vocab = build_artist_vocab(me, them)
    rows = build_track_raw_features(me, 0, vocab)
    assert all(len(r['behavioural']) == N_BEHAVIOURAL for r in rows)


def test_artist_block_is_a_one_hot(two_listeners):
    me, them = two_listeners
    vocab = build_artist_vocab(me, them)
    for row in build_track_raw_features(me, 0, vocab):
        onehot = row['artist_onehot']
        assert len(onehot) == len(vocab)
        assert onehot.sum() == 1.0
        assert onehot[vocab[row['artist']]] == 1.0


def test_tracks_by_different_artists_are_orthogonal(two_listeners):
    """
    The reason artist identity cannot separate listeners: two tracks by
    different artists are orthogonal whether or not the same person played
    them, so there is no co-occurrence structure for K-Means to find.
    """
    me, them = two_listeners
    vocab = build_artist_vocab(me, them)
    rows = build_track_raw_features(me, 0, vocab) + build_track_raw_features(them, 1, vocab)

    by_artist = {r['artist']: r['artist_onehot'] for r in rows}
    same_listener = float(np.dot(by_artist['Artist P'], by_artist['Artist Q']))
    across_listeners = float(np.dot(by_artist['Artist P'], by_artist['Artist X']))
    assert same_listener == across_listeners == 0.0


@pytest.mark.parametrize('use_artist', [True, False])
def test_cluster_tracks_reports_a_full_result(two_listeners, use_artist):
    me, them = two_listeners
    result = cluster_tracks(me, them, use_artist=use_artist)

    for key in ['ari', 'sil', 'cluster_labels', 'true_labels',
                'X_2d', 'outliers', 'cluster_stats', 'n_tracks']:
        assert key in result, key

    assert -1.0 <= result['ari'] <= 1.0
    assert -1.0 <= result['sil'] <= 1.0
    assert result['n_tracks'] == len(result['cluster_labels'])
    assert result['X_2d'].shape == (result['n_tracks'], 2)


def test_cluster_stats_account_for_every_track(two_listeners):
    me, them = two_listeners
    result = cluster_tracks(me, them)
    assert sum(s['total'] for s in result['cluster_stats']) == result['n_tracks']


def test_outliers_are_ranked_by_distance(two_listeners):
    me, them = two_listeners
    distances = cluster_tracks(me, them)['outliers']['distance']
    assert distances.is_monotonic_decreasing
