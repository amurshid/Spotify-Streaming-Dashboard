# Spotify Streaming Dashboard

An end-to-end machine learning pipeline over raw Spotify streaming history — engineered behavioral features, supervised preference classification, unsupervised listener clustering, and a similarity-based recommender — served through an interactive Plotly Dash app.

![Python](https://img.shields.io/badge/Python-3.8+-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange) ![Dash](https://img.shields.io/badge/Dash-4.x-lightblue) ![Plotly](https://img.shields.io/badge/Plotly-6.x-purple)

---

## Overview

Two people export their Spotify listening history. This project turns those raw JSON logs into a quantitative comparison of how they listen — and then asks a harder question: **can a model learn one person's taste well enough to pick songs for them out of someone else's library?**

The pipeline runs in four stages:

```
raw JSON  →  feature engineering  →  models  →  dashboard
             (dimensions.py)         (classifier, clustering,
                                       recommender)
```

---

## Machine Learning

### Feature engineering

Every user and every track is reduced to a set of normalized **behavioral probability vectors** ([`dimensions.py`](dimensions.py)):

| Vector | Dim | Captures |
|---|---|---|
| Hourly | 24 | Listening distribution across hour of day |
| Day of week | 7 | Weekday vs weekend rhythm |
| Seasonal | 4 | Winter / Spring / Summer / Fall |
| Listen duration | 6 | Bucketed play lengths (`<10s` → `3min+`) |
| Engagement | 3 | Skipped / partial / completed |
| Skip rate | 1 | Proportion of plays under 30s |
| Artist | sparse | Share of total minutes per artist |

Each vector is L1-normalized to a proportion, so users with wildly different total listening volume remain directly comparable.

### Similarity measure

[`similarity.py`](similarity.py) scores two listeners per dimension with cosine similarity, then combines them into a single weighted score. Weights renormalize over whichever dimensions are present, so the measure degrades gracefully on partial data.

### Supervised — would this listener like this track?

[`classifier.py`](classifier.py) builds one row per unique `(track, artist)` pair. Features are **similarity scores between that track's listening pattern and the user's overall profile** — the model never sees the track's raw identity, only how well its behavioral signature matches the listener.

Two classifiers are trained and compared on a stratified held-out split: a depth-limited **decision tree** and **KNN (k=5)**.

### Unsupervised — are two listeners separable?

[`clustering.py`](clustering.py) concatenates the 46-dim behavioral vector with a sparse artist one-hot, standardizes, then applies **PCA before K-Means**. The pre-reduction is deliberate: raw high-dimensional sparse one-hot encoding collapses Euclidean distance (curse of dimensionality) and K-Means degenerates. PCA compresses the artist structure into dense components the clustering can actually separate on.

Distance from each track to its assigned centroid doubles as an **outlier score**, surfacing the most behaviorally unusual tracks per listener.

### Recommender

[`recommender.py`](recommender.py) trains a model on one person's profile, then scores every track in the *other* person's library and ranks the predicted-liked tracks by `predict_proba` confidence.

---

## Methodology notes

Two decisions that are easy to get wrong, and how they're handled here:

**Label design and leakage.** Spotify's export contains no explicit ratings, so preference is inferred from implicit feedback: `liked = play_count > 1`. Because the label is derived from play count, the `is_repeat` feature is *identical to the label by construction*. It is computed for inspection but deliberately excluded from `FEATURES` — training on it would produce a model that reads the answer off its own input and reports a meaningless accuracy.

**Evaluation beyond accuracy.** Accuracy alone is uninformative on an imbalanced label. The pipeline reports:

- **Classification** — confusion matrix, precision and recall, and the tree's top split feature
- **Clustering** — Adjusted Rand Index against true listener identity, silhouette score, and per-cluster purity

Both are surfaced in the dashboard, not just printed, so the model's failure modes are visible alongside its successes.

---

## Dashboard

| Tab | Contents |
|---|---|
| 📊 Distribution | Histogram of track listening durations |
| 📈 Timeline | Daily listening time with 7-day rolling average |
| 🔥 Heatmap | Activity by hour of day vs day of week |
| 📦 Box Plots | Duration broken down by hour and weekday |
| 🎤 Artists / 🎵 Tracks | Ranked bar charts and word clouds sized by listening time |
| ⚡ Deep Dive | Parallel coordinates and an animated cumulative artist race |
| 🎵 Recommender | Live model switching (Decision Tree / KNN) with confidence-ranked picks and a confusion matrix |
| 🔬 Analytics | Similarity breakdown, PCA cluster projection, outlier tracks |

---

## Project Structure

```
├── app.py            # Dash entry point — server, recommender callbacks
├── main.py           # CLI entry point — runs the full analysis pipeline
│
├── ML / analysis
│   ├── dimensions.py   # Behavioral feature vectors (user- and track-level)
│   ├── similarity.py   # Cosine similarity + weighted scoring
│   ├── classifier.py   # Decision tree & KNN preference classifiers
│   ├── clustering.py   # PCA + K-Means, outlier detection, ARI/silhouette
│   ├── recommender.py  # Cross-library recommendation generation
│   └── analytics.py    # Precomputed similarity & clustering for the dashboard
│
├── data
│   ├── data_loader.py  # JSON loading, overlap-window alignment (CLI)
│   └── data.py         # Loading, derived columns, stats, word clouds (dashboard)
│
├── presentation
│   ├── charts.py       # Plotly figure builders
│   ├── layout.py       # Dash components and page layout
│   ├── config.py       # Theme constants and figure styling helpers
│   └── assets/style.css
```

---

## Getting Started

### 1. Get the data

Request your export from [Spotify Privacy Settings](https://www.spotify.com/account/privacy/), then place each person's `StreamingHistory_music_*.json` files in their own directory:

```
my_spotify_data/
├── StreamingHistory_music_0.json
└── StreamingHistory_music_1.json
atharva_more_spotify_data/
├── StreamingHistory_music_0.json
└── StreamingHistory_music_1.json
```

Both directories are gitignored. The exact filenames read are listed at the top of [`data.py`](data.py) and [`main.py`](main.py) — adjust them to match how many files your export contains.

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib plotly dash dash-bootstrap-components wordcloud
```

### 3. Run

```bash
python app.py    # interactive dashboard → http://127.0.0.1:8050
python main.py   # full analysis pipeline, printed to stdout
```

---

## Data Schema

Each record in a `StreamingHistory` file provides:

| Field | Description |
|---|---|
| `endTime` | Timestamp of when the track ended |
| `artistName` | Artist name |
| `trackName` | Track name |
| `msPlayed` | Milliseconds played |

Plays under 30 seconds are treated as skips throughout — Spotify's own threshold for counting a stream.

---

## Built With

- [scikit-learn](https://scikit-learn.org/) — Classification, clustering, PCA, metrics
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — Data processing and vector math
- [Dash](https://dash.plotly.com/) — Web framework for Python
- [Plotly](https://plotly.com/python/) — Interactive charting
- [Dash Bootstrap Components](https://dash-bootstrap-components.opensource.faculty.ai/) — UI components
- [WordCloud](https://github.com/amueller/word_cloud) — Word cloud generation
