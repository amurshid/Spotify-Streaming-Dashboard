import json
import pandas as pd


def load_streaming_history(paths):
    records = []
    for path in paths:
        records += json.load(open(path, encoding='utf-8'))
    df = pd.DataFrame(records)
    df['endTime'] = pd.to_datetime(df['endTime'])
    df['msPlayed'] = df['msPlayed'].astype(float)
    df['minutesPlayed'] = df['msPlayed'] / 60000
    return df


def restrict_to_overlap(df1, df2):
    start = max(df1['endTime'].min(), df2['endTime'].min())
    end   = min(df1['endTime'].max(), df2['endTime'].max())
    mask1 = (df1['endTime'] >= start) & (df1['endTime'] <= end)
    mask2 = (df2['endTime'] >= start) & (df2['endTime'] <= end)
    print(f"Overlapping window: {start.date()} to {end.date()}")
    return df1[mask1].copy(), df2[mask2].copy()
