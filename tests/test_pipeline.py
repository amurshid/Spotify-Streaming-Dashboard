"""
End-to-end smoke tests.

These are the cheap ones that would have caught the two crashes this project
shipped: main.py unpacking two values from a three-value return, and a
confusion matrix built without explicit labels raising on a single-class split.
Neither needed clever testing — only for something to run the path once.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from classifier import (build_user_profile, build_track_features, temporal_split,
                        train_classifier, predict_liked, predict_random_song)

ROOT = Path(__file__).resolve().parent.parent
REQUIRED_FIELDS = {'endTime', 'artistName', 'trackName', 'msPlayed'}


def test_full_training_path_runs(training_plays):
    """profile -> features -> train -> recommend, the way main.py drives it."""
    history, _, _ = temporal_split(training_plays)
    profile = build_user_profile(history)
    features = build_track_features(training_plays, profile)

    model, metrics = train_classifier(features)

    recommendations = predict_liked(model, training_plays, profile)
    assert set(recommendations.columns) == {'track', 'artist'}
    assert len(recommendations) <= training_plays['trackName'].nunique()

    track, artist, prediction = predict_random_song(model, training_plays, profile)
    assert prediction in (0, 1)
    assert isinstance(track, str) and isinstance(artist, str)


def test_trainers_return_three_way_unpackable_results(training_plays):
    """main.py unpacks these; a change in arity breaks it at runtime."""
    features = build_track_features(training_plays)
    model, metrics = train_classifier(features)
    assert model is not None and isinstance(metrics, dict)


def test_generator_emits_the_documented_schema(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run([sys.executable, str(ROOT / 'generate_sample_data.py')],
                   check=True, capture_output=True)

    files = sorted(tmp_path.glob('*/StreamingHistory_music_*.json'))
    assert files, 'generator produced no streaming history'

    for path in files:
        records = json.loads(path.read_text(encoding='utf-8'))
        assert records
        assert REQUIRED_FIELDS.issubset(records[0])
        assert all(isinstance(r['msPlayed'], int) for r in records[:50])


def test_generator_is_deterministic(tmp_path, monkeypatch):
    """Seeded, so a fixture-backed test run is reproducible."""
    outputs = []
    for run in ('first', 'second'):
        target = tmp_path / run
        target.mkdir()
        monkeypatch.chdir(target)
        subprocess.run([sys.executable, str(ROOT / 'generate_sample_data.py')],
                       check=True, capture_output=True)
        path = next(target.glob('*/StreamingHistory_music_0.json'))
        outputs.append(path.read_text(encoding='utf-8'))

    assert outputs[0] == outputs[1]
