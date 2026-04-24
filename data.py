import json
import io
import base64
import pandas as pd
from wordcloud import WordCloud

def _load_multi(paths):
    records = []
    for p in paths:
        records += json.load(open(p, encoding="utf-8"))
    return pd.DataFrame(records)

p1 = _load_multi([
    "my_spotify_data/StreamingHistory_music_0.json",
    "my_spotify_data/StreamingHistory_music_1.json",
    "my_spotify_data/StreamingHistory_music_2.json",
    "my_spotify_data/StreamingHistory_music_3.json",
    "my_spotify_data/StreamingHistory_music_4.json",
])
p2 = _load_multi([
    "atharva_more_spotify_data/StreamingHistory_music_0.json",
    "atharva_more_spotify_data/StreamingHistory_music_1.json",
])

for df in (p1, p2):
    df["endTime"]   = pd.to_datetime(df["endTime"])
    df["minPlayed"] = df["msPlayed"] / 60_000
    df["hour"]      = df["endTime"].dt.hour
    df["dayOfWeek"] = df["endTime"].dt.day_name()
    df["date"]      = df["endTime"].dt.date
    df["month"]     = df["endTime"].dt.to_period("M").astype(str)


def get_stats(df):
    return dict(
        streams    = f"{len(df):,}",
        hours      = f"{df['minPlayed'].sum()/60:,.0f} hrs",
        artists    = f"{df['artistName'].nunique():,}",
        tracks     = f"{df['trackName'].nunique():,}",
        top_artist = df.groupby("artistName")["minPlayed"].sum().idxmax(),
        top_track  = df.groupby("trackName" )["minPlayed"].sum().idxmax(),
        skip_pct   = f"{(df['msPlayed']<30_000).mean()*100:.1f}%",
        avg_min    = f"{df['minPlayed'].mean():.1f} min",
        date_range = f"{df['endTime'].min().strftime('%b %Y')} – {df['endTime'].max().strftime('%b %Y')}",
    )


S1, S2 = get_stats(p1), get_stats(p2)


def make_wc(df, col, cmap):
    freq = df.groupby(col)["minPlayed"].sum().to_dict()
    wc   = WordCloud(width=960, height=420, background_color="#0D0D0D",
                     colormap=cmap, max_words=90,
                     prefer_horizontal=0.85).generate_from_frequencies(freq)
    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


print("Generating word clouds…")
WC = dict(
    a1=make_wc(p1, "artistName", "Greens"), a2=make_wc(p2, "artistName", "Reds"),
    t1=make_wc(p1, "trackName",  "Greens"), t2=make_wc(p2, "trackName",  "Reds"),
)
