import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from wordcloud import WordCloud
import warnings
warnings.filterwarnings("ignore")

# ── Load data ──────────────────────────────────────────────────────────────────
p1 = pd.DataFrame(json.load(open("StreamingHistory_music_0.json", encoding="utf-8")))
p2 = pd.DataFrame(json.load(open("StreamingHistory0.json", encoding="utf-8")))

for df in (p1, p2):
    df["endTime"]   = pd.to_datetime(df["endTime"])
    df["minPlayed"] = df["msPlayed"] / 60_000
    df["hour"]      = df["endTime"].dt.hour
    df["dayOfWeek"] = df["endTime"].dt.day_name()
    df["date"]      = df["endTime"].dt.date

DOW_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
COLORS    = ["#1DB954", "#FF4B4B"]   # Spotify green vs red
C1, C2    = COLORS

# ── Helper ─────────────────────────────────────────────────────────────────────
def top_n(df, col, n=10):
    return df.groupby(col)["minPlayed"].sum().nlargest(n)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 – Histograms: listening-duration distribution
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Fig 1 – Distribution of Track Listening Duration", fontsize=14, fontweight="bold")

for ax, df, color, label in zip(axes, (p1, p2), COLORS, ("Person 1", "Person 2")):
    clipped = df["minPlayed"].clip(upper=10)
    ax.hist(clipped, bins=40, color=color, edgecolor="white", alpha=0.85)
    ax.set_title(label)
    ax.set_xlabel("Minutes Played (capped at 10 min)")
    ax.set_ylabel("Number of Streams")
    median = clipped.median()
    ax.axvline(median, color="black", linestyle="--", linewidth=1.5, label=f"Median {median:.1f} min")
    ax.legend()

plt.tight_layout()
plt.savefig("fig1_histogram.png", dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 – Line chart: daily total listening time
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
fig.suptitle("Fig 2 – Daily Total Listening Time", fontsize=14, fontweight="bold")

for ax, df, color, label in zip(axes, (p1, p2), COLORS, ("Person 1", "Person 2")):
    daily = df.groupby("date")["minPlayed"].sum().reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    rolling = daily.set_index("date")["minPlayed"].rolling("7D").mean()
    ax.fill_between(daily["date"], daily["minPlayed"], alpha=0.25, color=color)
    ax.plot(daily["date"], daily["minPlayed"], color=color, linewidth=0.8, alpha=0.7)
    ax.plot(rolling.index, rolling.values, color="black", linewidth=1.8, label="7-day rolling avg")
    ax.set_title(label)
    ax.set_ylabel("Minutes Played")
    ax.legend()

plt.tight_layout()
plt.savefig("fig2_linechart.png", dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 – Box plots: listening duration by hour of day
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Fig 3 – Listening Duration by Hour of Day (Box Plot)", fontsize=14, fontweight="bold")

for ax, df, color, label in zip(axes, (p1, p2), COLORS, ("Person 1", "Person 2")):
    groups = [df.loc[df["hour"] == h, "minPlayed"].clip(upper=10).values for h in range(24)]
    bp = ax.boxplot(groups, positions=range(24), widths=0.6, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2),
                    flierprops=dict(marker=".", markersize=2, alpha=0.4))
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(label)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Minutes Played (capped at 10)")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], rotation=45, fontsize=8)

plt.tight_layout()
plt.savefig("fig3_boxplot_hour.png", dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 – Box plots: listening duration by day of week
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Fig 4 – Listening Duration by Day of Week (Box Plot)", fontsize=14, fontweight="bold")

for ax, df, color, label in zip(axes, (p1, p2), COLORS, ("Person 1", "Person 2")):
    groups = [df.loc[df["dayOfWeek"] == d, "minPlayed"].clip(upper=10).values for d in DOW_ORDER]
    bp = ax.boxplot(groups, positions=range(7), widths=0.5, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2),
                    flierprops=dict(marker=".", markersize=2, alpha=0.4))
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(label)
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Minutes Played (capped at 10)")
    ax.set_xticks(range(7))
    ax.set_xticklabels([d[:3] for d in DOW_ORDER])

plt.tight_layout()
plt.savefig("fig4_boxplot_dow.png", dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 – Bar charts: Top 10 Artists by total listening time
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Fig 5 – Top 10 Artists by Total Listening Time", fontsize=14, fontweight="bold")

for ax, df, color, label in zip(axes, (p1, p2), COLORS, ("Person 1", "Person 2")):
    data = top_n(df, "artistName", 10).sort_values()
    bars = ax.barh(data.index, data.values, color=color, edgecolor="white", alpha=0.85)
    ax.bar_label(bars, fmt="%.0f min", padding=3, fontsize=8)
    ax.set_title(label)
    ax.set_xlabel("Total Minutes Played")

plt.tight_layout()
plt.savefig("fig5_top_artists.png", dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 – Bar charts: Top 10 Tracks by total listening time
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Fig 6 – Top 10 Tracks by Total Listening Time", fontsize=14, fontweight="bold")

for ax, df, color, label in zip(axes, (p1, p2), COLORS, ("Person 1", "Person 2")):
    data = top_n(df, "trackName", 10).sort_values()
    short = [t if len(t) <= 40 else t[:37] + "..." for t in data.index]
    bars = ax.barh(short, data.values, color=color, edgecolor="white", alpha=0.85)
    ax.bar_label(bars, fmt="%.0f min", padding=3, fontsize=8)
    ax.set_title(label)
    ax.set_xlabel("Total Minutes Played")

plt.tight_layout()
plt.savefig("fig6_top_tracks.png", dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 – Word Clouds: Artists
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Fig 7 – Artist Word Cloud (size = total listening time)", fontsize=14, fontweight="bold")

for ax, df, cmap, label in zip(axes, (p1, p2), ("Greens", "Reds"), ("Person 1", "Person 2")):
    freq = df.groupby("artistName")["minPlayed"].sum().to_dict()
    wc = WordCloud(width=800, height=500, background_color="white",
                   colormap=cmap, max_words=80).generate_from_frequencies(freq)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(label, fontsize=12)

plt.tight_layout()
plt.savefig("fig7_wordcloud_artists.png", dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 – Word Clouds: Tracks
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Fig 8 – Track Word Cloud (size = total listening time)", fontsize=14, fontweight="bold")

for ax, df, cmap, label in zip(axes, (p1, p2), ("Greens", "Reds"), ("Person 1", "Person 2")):
    freq = df.groupby("trackName")["minPlayed"].sum().to_dict()
    wc = WordCloud(width=800, height=500, background_color="white",
                   colormap=cmap, max_words=80).generate_from_frequencies(freq)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(label, fontsize=12)

plt.tight_layout()
plt.savefig("fig8_wordcloud_tracks.png", dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9 – Heatmap: Listening activity (hour vs day of week)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Fig 9 – Listening Activity Heatmap (Hour vs Day of Week)", fontsize=14, fontweight="bold")

for ax, df, cmap, label in zip(axes, (p1, p2), ("Greens", "Reds"), ("Person 1", "Person 2")):
    heat = df.groupby(["dayOfWeek", "hour"])["minPlayed"].sum().unstack(fill_value=0)
    heat = heat.reindex(DOW_ORDER)
    heat = heat.reindex(columns=range(24), fill_value=0)
    im = ax.imshow(heat.values, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_title(label)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], rotation=45, fontsize=7)
    ax.set_yticks(range(7))
    ax.set_yticklabels([d[:3] for d in DOW_ORDER])
    plt.colorbar(im, ax=ax, label="Total Minutes")

plt.tight_layout()
plt.savefig("fig9_heatmap.png", dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 10 – Parallel Coordinates: Top artists comparison
# ══════════════════════════════════════════════════════════════════════════════
def artist_profile(df, top_artists):
    rows = []
    for artist in top_artists:
        sub = df[df["artistName"] == artist]
        rows.append({
            "artist":    artist[:25],
            "total_min": sub["minPlayed"].sum(),
            "streams":   len(sub),
            "avg_min":   sub["minPlayed"].mean(),
            "skip_rate": (sub["msPlayed"] < 30_000).mean() * 100,
        })
    return pd.DataFrame(rows)

top15_p1 = p1.groupby("artistName")["minPlayed"].sum().nlargest(15).index.tolist()
top15_p2 = p2.groupby("artistName")["minPlayed"].sum().nlargest(15).index.tolist()
prof1 = artist_profile(p1, top15_p1)
prof2 = artist_profile(p2, top15_p2)

AXES_COLS   = ["total_min", "streams", "avg_min", "skip_rate"]
AXES_LABELS = ["Total\nMinutes", "Stream\nCount", "Avg Session\n(min)", "Skip\nRate (%)"]

fig, ax = plt.subplots(figsize=(14, 7))
fig.suptitle("Fig 10 – Parallel Coordinates: Top-15 Artists per Person", fontsize=14, fontweight="bold")

def normalize(series):
    mn, mx = series.min(), series.max()
    return (series - mn) / (mx - mn + 1e-9)

for prof, color, label in [(prof1, C1, "Person 1"), (prof2, C2, "Person 2")]:
    normed = prof[AXES_COLS].apply(normalize)
    for _, row in normed.iterrows():
        ax.plot(range(len(AXES_COLS)), row.values, color=color, alpha=0.55, linewidth=1.2)

ax.set_xticks(range(len(AXES_COLS)))
ax.set_xticklabels(AXES_LABELS, fontsize=10)
ax.set_ylabel("Normalized Value (0–1)")
ax.set_ylim(-0.05, 1.05)
for x in range(len(AXES_COLS)):
    ax.axvline(x, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
legend_handles = [Patch(color=C1, label="Person 1"), Patch(color=C2, label="Person 2")]
ax.legend(handles=legend_handles, loc="upper right")

plt.tight_layout()
plt.savefig("fig10_parallel_coord.png", dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 11 – Grouped bar: hourly listening volume comparison
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 5))
fig.suptitle("Fig 11 – Hourly Listening Volume Comparison", fontsize=14, fontweight="bold")

hours = np.arange(24)
h1 = p1.groupby("hour")["minPlayed"].sum().reindex(range(24), fill_value=0)
h2 = p2.groupby("hour")["minPlayed"].sum().reindex(range(24), fill_value=0)

width = 0.4
ax.bar(hours - width/2, h1.values, width=width, color=C1, alpha=0.85, label="Person 1")
ax.bar(hours + width/2, h2.values, width=width, color=C2, alpha=0.85, label="Person 2")
ax.set_xticks(hours)
ax.set_xticklabels([f"{h:02d}:00" for h in hours], rotation=45, fontsize=8)
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Total Minutes Played")
ax.legend()

plt.tight_layout()
plt.savefig("fig11_hourly_bar.png", dpi=150)
plt.show()

print("All 11 figures saved.")
