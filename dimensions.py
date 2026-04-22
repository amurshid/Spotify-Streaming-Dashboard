import numpy as np


def hourly_vector(df):
    """24-dim proportion vector of listening by hour of day."""
    df = df.copy()
    df['hour'] = df['endTime'].dt.hour
    hourly = df.groupby('hour')['minutesPlayed'].sum()
    vec = np.array([hourly.get(h, 0.0) for h in range(24)])
    return vec / vec.sum() if vec.sum() > 0 else vec


def artist_vector(df):
    """Dict of {artist: proportion of total minutes}."""
    artist_mins = df.groupby('artistName')['minutesPlayed'].sum()
    total = artist_mins.sum()
    return (artist_mins / total).to_dict() if total > 0 else {}


def _get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    if month in [3, 4, 5]:  return 'Spring'
    if month in [6, 7, 8]:  return 'Summer'
    return 'Fall'

SEASONS = ['Winter', 'Spring', 'Summer', 'Fall']

def seasonal_vector(df):
    """4-dim proportion vector of listening by season."""
    df = df.copy()
    df['season'] = df['endTime'].dt.month.map(_get_season)
    season_mins = df.groupby('season')['minutesPlayed'].sum()
    vec = np.array([season_mins.get(s, 0.0) for s in SEASONS])
    return vec / vec.sum() if vec.sum() > 0 else vec


DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

def dow_vector(df):
    """7-dim proportion vector of listening by day of week."""
    df = df.copy()
    df['dow'] = df['endTime'].dt.day_name()
    dow_mins = df.groupby('dow')['minutesPlayed'].sum()
    vec = np.array([dow_mins.get(d, 0.0) for d in DAYS])
    return vec / vec.sum() if vec.sum() > 0 else vec


def skip_rate(df):
    """Proportion of tracks played under 30 seconds (skipped)."""
    skipped = (df['msPlayed'] < 30000).sum()
    return skipped / len(df) if len(df) > 0 else 0.0


def track_diversity(df):
    """Unique tracks / total plays — how much does the user repeat songs."""
    return df['trackName'].nunique() / len(df) if len(df) > 0 else 0.0


# --- Per-track vectors (input: a grouped subset of plays for one track) ---

def track_hourly_vector(group):
    """24-dim proportion vector for a single track's plays by hour."""
    counts = group['endTime'].dt.hour.value_counts()
    vec = np.array([counts.get(h, 0.0) for h in range(24)], dtype=float)
    return vec / vec.sum() if vec.sum() > 0 else vec


def track_seasonal_vector(group):
    """4-dim proportion vector for a single track's plays by season."""
    seasons = group['endTime'].dt.month.map(_get_season)
    counts = seasons.value_counts()
    vec = np.array([counts.get(s, 0.0) for s in SEASONS], dtype=float)
    return vec / vec.sum() if vec.sum() > 0 else vec


def track_dow_vector(group):
    """7-dim proportion vector for a single track's plays by day of week."""
    days = group['endTime'].dt.day_name()
    counts = days.value_counts()
    vec = np.array([counts.get(d, 0.0) for d in DAYS], dtype=float)
    return vec / vec.sum() if vec.sum() > 0 else vec


def track_skip_rate(group):
    """Fraction of plays under 30s for a specific track group."""
    return float((group['msPlayed'] < 30000).sum()) / len(group) if len(group) > 0 else 0.0


# --- Listen duration similarity (mirrors hourly/seasonal pattern approach) ---

DURATION_BINS   = [0, 10_000, 30_000, 60_000, 120_000, 180_000, float('inf')]
DURATION_LABELS = ['<10s', '10-30s', '30-60s', '1-2min', '2-3min', '3min+']


def _duration_bucket(ms):
    for i, edge in enumerate(DURATION_BINS[1:]):
        if ms < edge:
            return DURATION_LABELS[i]
    return DURATION_LABELS[-1]


def listen_duration_vector(df):
    """6-dim proportion vector of all plays bucketed by how long they were listened to."""
    buckets = df['msPlayed'].apply(_duration_bucket)
    counts  = buckets.value_counts()
    vec = np.array([counts.get(b, 0.0) for b in DURATION_LABELS], dtype=float)
    return vec / vec.sum() if vec.sum() > 0 else vec


def track_listen_duration_vector(group):
    """6-dim proportion vector for a single track's plays by listen duration bucket."""
    buckets = group['msPlayed'].apply(_duration_bucket)
    counts  = buckets.value_counts()
    vec = np.array([counts.get(b, 0.0) for b in DURATION_LABELS], dtype=float)
    return vec / vec.sum() if vec.sum() > 0 else vec


# --- Engagement similarity (skip / partial / complete breakdown) ---

def engagement_vector(df):
    """3-dim proportion vector: [skipped (<30s), partial (30s-2min), completed (2min+)]."""
    total = len(df)
    if total == 0:
        return np.zeros(3)
    skipped  = (df['msPlayed'] < 30_000).sum()
    complete = (df['msPlayed'] >= 120_000).sum()
    partial  = total - skipped - complete
    vec = np.array([skipped, partial, complete], dtype=float)
    return vec / vec.sum() if vec.sum() > 0 else vec


def track_engagement_vector(group):
    """3-dim engagement vector [skipped, partial, completed] for a single track's plays."""
    total = len(group)
    if total == 0:
        return np.zeros(3)
    skipped  = (group['msPlayed'] < 30_000).sum()
    complete = (group['msPlayed'] >= 120_000).sum()
    partial  = total - skipped - complete
    vec = np.array([skipped, partial, complete], dtype=float)
    return vec / vec.sum() if vec.sum() > 0 else vec
