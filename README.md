# Spotify Streaming Dashboard

An interactive web dashboard comparing the Spotify listening habits of two people, built with Plotly and Dash.

![Python](https://img.shields.io/badge/Python-3.8+-blue) ![Dash](https://img.shields.io/badge/Dash-4.x-lightblue) ![Plotly](https://img.shields.io/badge/Plotly-6.x-purple)

---

## Overview

This project takes raw Spotify streaming history JSON exports and visualizes them as an interactive dark-themed dashboard. It compares listening patterns, top artists, top tracks, and behavioral habits across two individuals.

---

## Features

- **Distribution** — Histogram of track listening durations
- **Timeline** — Daily listening time with 7-day rolling average
- **Heatmap** — Listening activity by hour of day vs day of week
- **Box Plots** — Listening duration broken down by hour and day of week
- **Top Artists & Tracks** — Horizontal bar charts with gradient coloring
- **Word Clouds** — Artist and track word clouds sized by total listening time
- **Parallel Coordinates** — Multi-metric artist profile comparison
- **Animated Bar Race** — Cumulative top artists race animated by month

---

## Project Structure

```
├── app.py          # Entry point — initializes Dash app and runs server
├── config.py       # Constants: colors, theme helpers (dark, grad, rgba)
├── data.py         # Data loading, preprocessing, stats, word cloud generation
├── charts.py       # All Plotly figure builder functions
├── layout.py       # Dash component builders and final app layout
├── assets/
│   └── style.css   # Custom CSS (auto-served by Dash)
└── README.md
```

---

## Getting Started

### 1. Get your Spotify data

Request your data from [Spotify Privacy Settings](https://www.spotify.com/account/privacy/). Once downloaded, place these files in the project root:

```
StreamingHistory_music_0.json
StreamingHistory0.json
```

### 2. Install dependencies

```bash
pip install pandas plotly dash dash-bootstrap-components wordcloud
```

### 3. Run the dashboard

```bash
python app.py
```

Then open your browser at `http://127.0.0.1:8050`

---

## Data

The dashboard expects two Spotify `StreamingHistory` JSON files. Each record should contain:

| Field | Description |
|---|---|
| `endTime` | Timestamp of when the track ended |
| `artistName` | Artist name |
| `trackName` | Track name |
| `msPlayed` | Milliseconds played |

---

## Built With

- [Dash](https://dash.plotly.com/) — Web framework for Python
- [Plotly](https://plotly.com/python/) — Interactive charting
- [Dash Bootstrap Components](https://dash-bootstrap-components.opensource.faculty.ai/) — UI components
- [WordCloud](https://github.com/amueller/word_cloud) — Word cloud generation
- [Pandas](https://pandas.pydata.org/) — Data processing
