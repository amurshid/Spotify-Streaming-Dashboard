"""
Generate synthetic Spotify streaming exports as a development fixture.

Spotify takes several days to fulfil a data export, so this exists to exercise
the pipeline end to end in the meantime. It is a smoke test, not a demo: the
synthetic data has no real relationship between a track's history and whether
it is returned to later, so model scores on it sit near chance. It verifies
that the code runs, not that the model works.

    python generate_sample_data.py
"""
import json
import os
import random
from datetime import datetime, timedelta

SEED = 42
DAYS = 730
OUT = {
    'my_spotify_data':           {'files': 5, 'tracks': 3_300, 'profile': 'night_owl'},
    'atharva_more_spotify_data': {'files': 2, 'tracks': 1_300, 'profile': 'daytime'},
}

# Hour-of-day weights. The two profiles overlap in the evening but diverge
# sharply overnight and mid-morning, which is what the hourly cosine
# similarity is meant to pick up.
HOUR_WEIGHTS = {
    'night_owl': [8, 6, 4, 2, 1, 1, 1, 1, 2, 3, 3, 4,
                  4, 4, 4, 5, 5, 6, 7, 8, 9, 10, 10, 9],
    'daytime':   [1, 1, 1, 1, 1, 2, 4, 7, 9, 10, 10, 9,
                  8, 8, 9, 9, 8, 7, 6, 5, 4, 3, 2, 1],
}

# Day-of-week weights, Monday first. The daytime listener is weekend-heavy.
DOW_WEIGHTS = {
    'night_owl': [5, 5, 5, 5, 6, 6, 5],
    'daytime':   [4, 4, 4, 4, 6, 9, 9],
}

SHARED_ARTISTS = [
    'Mount Avalon', 'The Paper Kites Society', 'Novembre', 'Glass Harbour',
    'Sundial Motel', 'Kite & Anchor',
]
ARTISTS = {
    'night_owl': SHARED_ARTISTS + [
        'Low Tide Choir', 'Ashgrove', 'Velvet Ceiling', 'Midnight Cartography',
        'Ivory Lanes', 'The Quiet Hour', 'Static Bloom', 'Hollow Pines',
        'Neon Orchard', 'Saltwater Diary', 'Empty Orchestra', 'Blue Hour Radio',
        'Paper Lantern', 'Drifting North', 'Cedar & Smoke',
    ],
    'daytime': SHARED_ARTISTS + [
        'Bright Antenna', 'Sunroom', 'The Commuters', 'Portside',
        'Golden Arcade', 'Fair Weather Club', 'Open Signal', 'Marigold Ave',
        'Tuesday Morning', 'Wide Field', 'The Lighthouse Keepers', 'Copper Lake',
        'Harbourtown', 'First Light', 'Summerhouse',
    ],
}

TRACK_WORDS_A = [
    'Amber', 'Quiet', 'Slow', 'Northern', 'Paper', 'Velvet', 'Golden', 'Hollow',
    'Silver', 'Winter', 'Open', 'Distant', 'Bright', 'Falling', 'Endless',
    'Crooked', 'Patient', 'Hidden', 'Narrow', 'Restless', 'Shallow', 'Steady',
    'Uneven', 'Faded', 'Certain', 'Borrowed', 'Kindred', 'Late', 'Second',
    'Thirteen',
]
TRACK_WORDS_B = [
    'Lights', 'Hours', 'Weather', 'Rooms', 'Summer', 'Rivers', 'Tides', 'Avenue',
    'Signal', 'Mornings', 'Harbour', 'Static', 'Ceiling', 'Motion', 'Lines',
    'Corners', 'Windows', 'Gardens', 'Letters', 'Traffic', 'Fever', 'Ladders',
    'Islands', 'Machines', 'Patterns', 'Seasons', 'Telephone', 'Velocity',
    'Wilderness', 'Yards',
]

# Distribution of how many times a track is played across the whole history.
# Roughly 45% of tracks are heard exactly once, which is what keeps the
# "liked = played more than once" label from collapsing to a single class.
PLAY_COUNTS = [1, 2, 3, 4, 5, 8, 12, 20, 35]
PLAY_WEIGHTS = [45, 18, 10, 7, 5, 7, 4, 3, 1]


def _build_library(rng, profile, n_tracks):
    """Assign each artist a slice of the track pool and a play count per track.

    Play counts follow a long tail (see PLAY_COUNTS): a handful of tracks
    dominate the history while most are heard once, which is what gives the
    classifier a non-trivial label distribution.
    """
    artists = ARTISTS[profile]
    per_artist = n_tracks // len(artists)
    pool = [f'{a} {b}' for a in TRACK_WORDS_A for b in TRACK_WORDS_B]

    library = []
    for artist in artists:
        for track in rng.sample(pool, per_artist):
            plays = rng.choices(PLAY_COUNTS, weights=PLAY_WEIGHTS, k=1)[0]
            library.append((track, artist, plays))
    return library


def _pick_timestamp(rng, profile, start):
    day = rng.choices(range(DAYS), k=1)[0]
    date = start + timedelta(days=day)
    dow_weight = DOW_WEIGHTS[profile][date.weekday()]

    # Resample the day until it lands on a weekday the listener actually
    # listens on, so the day-of-week distribution matches the profile.
    while rng.random() > dow_weight / max(DOW_WEIGHTS[profile]):
        day = rng.randrange(DAYS)
        date = start + timedelta(days=day)
        dow_weight = DOW_WEIGHTS[profile][date.weekday()]

    hour = rng.choices(range(24), weights=HOUR_WEIGHTS[profile], k=1)[0]
    return date.replace(hour=hour, minute=rng.randrange(60))


def _generate_plays(rng, profile, n_tracks):
    library = _build_library(rng, profile, n_tracks)
    start   = datetime(2023, 1, 1)

    records = []
    for track, artist, plays in library:
        for _ in range(plays):
            end = _pick_timestamp(rng, profile, start)

            if rng.random() < 0.18:
                ms = rng.randrange(1_000, 30_000)      # skipped
            else:
                ms = rng.randrange(95_000, 310_000)    # played through

            records.append({
                'endTime':    end.strftime('%Y-%m-%d %H:%M'),
                'artistName': artist,
                'trackName':  track,
                'msPlayed':   ms,
            })

    records.sort(key=lambda r: r['endTime'])
    return records


def main():
    rng = random.Random(SEED)

    for directory, spec in OUT.items():
        os.makedirs(directory, exist_ok=True)
        records = _generate_plays(rng, spec['profile'], spec['tracks'])

        chunk = len(records) // spec['files']
        for i in range(spec['files']):
            lo = i * chunk
            hi = len(records) if i == spec['files'] - 1 else (i + 1) * chunk
            path = os.path.join(directory, f'StreamingHistory_music_{i}.json')
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(records[lo:hi], f, ensure_ascii=False, indent=1)

        tracks = len({(r['trackName'], r['artistName']) for r in records})
        print(f"{directory:<28} {len(records):>6,} plays  "
              f"{tracks:>4} unique tracks  →  {spec['files']} file(s)")


if __name__ == '__main__':
    main()
