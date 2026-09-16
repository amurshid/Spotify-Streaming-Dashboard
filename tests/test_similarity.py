"""Tests for the similarity measure and its weighted aggregation."""
import numpy as np
import pytest

from similarity import cosine_sim, scalar_sim, compute_final_score, WEIGHTS


def test_identical_vectors_score_one():
    assert cosine_sim([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_orthogonal_vectors_score_zero():
    assert cosine_sim([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_scale_does_not_affect_cosine():
    assert cosine_sim([1.0, 2.0], [10.0, 20.0]) == pytest.approx(1.0)


def test_zero_vector_scores_zero_rather_than_dividing_by_zero():
    assert cosine_sim([0.0, 0.0], [1.0, 2.0]) == 0.0
    assert cosine_sim([0.0, 0.0], [0.0, 0.0]) == 0.0


def test_dicts_are_aligned_on_their_union_of_keys():
    assert cosine_sim({'a': 1.0}, {'b': 1.0}) == pytest.approx(0.0)
    assert cosine_sim({'a': 1.0, 'b': 1.0}, {'a': 1.0, 'b': 1.0}) == pytest.approx(1.0)


def test_disjoint_artist_profiles_are_dissimilar():
    assert cosine_sim({'X': 0.6, 'Y': 0.4}, {'P': 0.5, 'Q': 0.5}) == pytest.approx(0.0)


def test_scalar_similarity_is_one_minus_distance():
    assert scalar_sim(0.3, 0.3) == pytest.approx(1.0)
    assert scalar_sim(0.0, 1.0) == pytest.approx(0.0)


def test_final_score_renormalises_over_present_dimensions():
    """A subset of dimensions still yields a score on the same [0, 1] scale."""
    assert compute_final_score({'artist': 1.0, 'hourly': 1.0}) == pytest.approx(1.0)
    assert compute_final_score({'artist': 0.0, 'hourly': 0.0}) == pytest.approx(0.0)


def test_final_score_respects_relative_weights():
    """artist carries more weight than hourly, so scoring on it pulls higher."""
    assert WEIGHTS['artist'] > WEIGHTS['hourly']
    artist_strong = compute_final_score({'artist': 1.0, 'hourly': 0.0})
    hourly_strong = compute_final_score({'artist': 0.0, 'hourly': 1.0})
    assert artist_strong > hourly_strong


def test_final_score_of_nothing_is_zero():
    assert compute_final_score({}) == 0.0
