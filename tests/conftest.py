import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COMPLETE_MS = 200_000
SKIP_MS = 5_000


def make_plays(rows):
    """
    Build a streaming-history frame from (track, artist, when, ms) tuples,
    with the derived columns the pipeline expects.
    """
    df = pd.DataFrame(rows, columns=['trackName', 'artistName', 'endTime', 'msPlayed'])
    df['endTime'] = pd.to_datetime(df['endTime'])
    df['msPlayed'] = df['msPlayed'].astype(float)
    df['minutesPlayed'] = df['msPlayed'] / 60_000
    return df.sort_values('endTime').reset_index(drop=True)


@pytest.fixture
def simple_plays():
    """A handful of plays with hand-checkable statistics."""
    base = datetime(2024, 1, 1, 9, 0)
    return make_plays([
        ('Track A', 'Artist X', base,                        COMPLETE_MS),
        ('Track A', 'Artist X', base + timedelta(days=1),    COMPLETE_MS),
        ('Track B', 'Artist X', base + timedelta(days=2),    SKIP_MS),
        ('Track C', 'Artist Y', base + timedelta(days=3),    COMPLETE_MS),
    ])


@pytest.fixture
def empty_plays():
    return make_plays([])


@pytest.fixture
def training_plays():
    """
    A history large enough to train on: 160 tracks across the first 80% of the
    span, half of which are played again in the final 20%.

    Deliberately seeded and balanced so a stratified 5-fold split is always
    viable, and so tests assert on contracts rather than on model quality.
    """
    rng = np.random.default_rng(0)
    start = datetime(2024, 1, 1)
    rows = []

    for i in range(160):
        artist = f'Artist {i % 12}'
        track = f'Track {i}'
        for _ in range(int(rng.integers(1, 6))):
            day = int(rng.integers(0, 200))
            rows.append((track, artist, start + timedelta(days=day, hours=int(rng.integers(0, 24))),
                         COMPLETE_MS if rng.random() > 0.2 else SKIP_MS))
        # Half the tracks are returned to in the held-out window.
        if i % 2 == 0:
            rows.append((track, artist, start + timedelta(days=int(rng.integers(220, 260))),
                         COMPLETE_MS))

    return make_plays(rows)
