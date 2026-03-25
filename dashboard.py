import json, base64, io
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

# ── Load & enrich data ─────────────────────────────────────────────────────
p1 = pd.DataFrame(json.load(open("StreamingHistory_music_0.json", encoding="utf-8")))
p2 = pd.DataFrame(json.load(open("StreamingHistory0.json",        encoding="utf-8")))

for df in (p1, p2):
    df["endTime"]   = pd.to_datetime(df["endTime"])
    df["minPlayed"] = df["msPlayed"] / 60_000
    df["hour"]      = df["endTime"].dt.hour
    df["dayOfWeek"] = df["endTime"].dt.day_name()
    df["date"]      = df["endTime"].dt.date
    df["month"]     = df["endTime"].dt.to_period("M").astype(str)

DOW = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
C1, C2   = "#1DB954", "#FF6B6B"
BG       = "#0D0D0D"
CARD     = "#161616"
CARD2    = "#1E1E1E"
TEXT     = "#FFFFFF"
DIM      = "#A0A0A0"

# ── Colour helpers ─────────────────────────────────────────────────────────
def hex_rgb(h):
    h = h.lstrip("#")
    return int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)

def rgba(h, a): r,g,b = hex_rgb(h); return f"rgba({r},{g},{b},{a})"

def grad(color):
    r,g,b = hex_rgb(color)
    return [[0, f"rgba({r},{g},{b},0.25)"], [1, color]]

# ── Dark layout template ───────────────────────────────────────────────────
def dark(title="", height=420, legend=True):
    d = dict(
        title     = dict(text=title, font=dict(color=TEXT, size=14, family="Inter"),
                         x=0.01, xanchor="left"),
        height    = height,
        paper_bgcolor = CARD2,
        plot_bgcolor  = "#111111",
        font      = dict(color=DIM, family="Inter, sans-serif", size=11),
        margin    = dict(l=14, r=14, t=48, b=14),
        legend    = dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT)) if legend else None,
        transition= dict(duration=500, easing="cubic-in-out"),
        hoverlabel= dict(bgcolor="#1E1E1E", font_color=TEXT, bordercolor="#333"),
    )
    if not legend:
        d.pop("legend")
    return d

def grid_axes(fig, rows=1, cols=1):
    fig.update_xaxes(gridcolor="#252525", zerolinecolor="#333", linecolor="#333")
    fig.update_yaxes(gridcolor="#252525", zerolinecolor="#333", linecolor="#333")

# ── Word-cloud helper ──────────────────────────────────────────────────────
def make_wc(df, col, cmap):
    freq = df.groupby(col)["minPlayed"].sum().to_dict()
    wc   = WordCloud(width=960, height=420, background_color="#0D0D0D",
                     colormap=cmap, max_words=90,
                     prefer_horizontal=0.85).generate_from_frequencies(freq)
    buf  = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

print("Generating word clouds…")
WC = dict(
    a1=make_wc(p1, "artistName", "Greens"), a2=make_wc(p2, "artistName", "Reds"),
    t1=make_wc(p1, "trackName",  "Greens"), t2=make_wc(p2, "trackName",  "Reds"),
)

# ── Stats ──────────────────────────────────────────────────────────────────
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

# ══════════════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════

def fig_histogram():
    fig = make_subplots(1, 2, subplot_titles=["Person 1  (2025)", "Person 2  (2020–21)"],
                        horizontal_spacing=0.08)
    for col, (df, color) in enumerate([(p1,C1),(p2,C2)], 1):
        clipped = df["minPlayed"].clip(upper=10)
        median  = clipped.median()
        fig.add_trace(go.Histogram(
            x=clipped, nbinsx=45, name=f"Person {col}",
            marker=dict(color=color, opacity=0.80, line=dict(width=0)),
            hovertemplate="~%{x:.1f} min<br>Count: %{y}<extra></extra>"), row=1, col=col)
        fig.add_vline(x=median, line_dash="dot", line_color="white", line_width=1.5,
                      annotation_text=f"med {median:.1f}m",
                      annotation_font=dict(color="white", size=10),
                      row=1, col=col)
    fig.update_layout(**dark("Track Duration Distribution (capped @ 10 min)"))
    grid_axes(fig)
    return fig


def fig_linechart():
    fig = make_subplots(2, 1, subplot_titles=["Person 1  (2025)", "Person 2  (2020–21)"],
                        shared_xaxes=False, vertical_spacing=0.14)
    for row, (df, color) in enumerate([(p1,C1),(p2,C2)], 1):
        daily = df.groupby("date")["minPlayed"].sum().reset_index()
        daily["date"] = pd.to_datetime(daily["date"])
        roll  = daily.set_index("date")["minPlayed"].rolling("7D").mean()
        fig.add_trace(go.Scatter(
            x=daily["date"], y=daily["minPlayed"], fill="tozeroy",
            fillcolor=rgba(color, 0.10), line=dict(color=color, width=0.9),
            name=f"P{row} daily",
            hovertemplate="%{x|%b %d, %Y}<br>%{y:.0f} min<extra></extra>"), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=roll.index, y=roll.values,
            line=dict(color="white", width=2),
            name=f"P{row} 7-day avg",
            hovertemplate="%{x|%b %d}<br>Avg %{y:.0f} min<extra></extra>"), row=row, col=1)
        fig.update_xaxes(
            rangeslider=dict(visible=(row == 2), bgcolor="#1A1A1A", thickness=0.05),
            gridcolor="#252525", linecolor="#333", row=row, col=1)
        fig.update_yaxes(gridcolor="#252525", linecolor="#333", row=row, col=1)
    fig.update_layout(**dark("Daily Listening Time with 7-day Rolling Average", height=540))
    return fig


def fig_heatmap():
    fig = make_subplots(1, 2, subplot_titles=["Person 1", "Person 2"],
                        horizontal_spacing=0.10)
    for col, (df, cmap) in enumerate([(p1,"Greens"),(p2,"Reds")], 1):
        heat = (df.groupby(["dayOfWeek","hour"])["minPlayed"]
                  .sum().unstack(fill_value=0)
                  .reindex(DOW)
                  .reindex(columns=range(24), fill_value=0))
        fig.add_trace(go.Heatmap(
            z=heat.values,
            x=[f"{h:02d}:00" for h in range(24)],
            y=[d[:3] for d in DOW],
            colorscale=cmap, showscale=True,
            hovertemplate="<b>%{y}  %{x}</b><br>%{z:.0f} min<extra></extra>"),
            row=1, col=col)
    fig.update_layout(**dark("Listening Activity Heatmap  ·  Hour × Day of Week"))
    fig.update_xaxes(tickfont=dict(size=9))
    return fig


def fig_box_hour():
    fig = make_subplots(1, 2, subplot_titles=["Person 1", "Person 2"],
                        horizontal_spacing=0.06)
    for col, (df, color) in enumerate([(p1,C1),(p2,C2)], 1):
        for h in range(24):
            vals = df.loc[df["hour"]==h, "minPlayed"].clip(upper=10).values
            fig.add_trace(go.Box(
                y=vals, name=f"{h:02d}",
                marker=dict(color=color, size=2, opacity=0.5),
                line=dict(color=color), boxmean=True,
                showlegend=False,
                hovertemplate=f"<b>{h:02d}:00</b><br>%{{y:.1f}} min<extra></extra>"),
                row=1, col=col)
    fig.update_layout(**dark("Listening Duration by Hour of Day  ·  Box Plots", height=440))
    grid_axes(fig)
    fig.update_xaxes(tickfont=dict(size=9))
    return fig


def fig_box_dow():
    fig = make_subplots(1, 2, subplot_titles=["Person 1", "Person 2"],
                        horizontal_spacing=0.06)
    for col, (df, color) in enumerate([(p1,C1),(p2,C2)], 1):
        for d in DOW:
            vals = df.loc[df["dayOfWeek"]==d, "minPlayed"].clip(upper=10).values
            fig.add_trace(go.Box(
                y=vals, name=d[:3],
                marker=dict(color=color, size=2, opacity=0.5),
                line=dict(color=color), boxmean=True,
                showlegend=False), row=1, col=col)
    fig.update_layout(**dark("Listening Duration by Day of Week  ·  Box Plots"))
    grid_axes(fig)
    return fig


def fig_top_artists():
    fig = make_subplots(1, 2, subplot_titles=["Person 1", "Person 2"],
                        horizontal_spacing=0.20)
    for col, (df, color) in enumerate([(p1,C1),(p2,C2)], 1):
        d = df.groupby("artistName")["minPlayed"].sum().nlargest(10).sort_values()
        fig.add_trace(go.Bar(
            x=d.values, y=d.index, orientation="h",
            marker=dict(color=d.values, colorscale=grad(color), showscale=False,
                        line=dict(width=0)),
            text=[f" {v:.0f}" for v in d.values], textposition="outside",
            textfont=dict(color=DIM, size=10), showlegend=False,
            hovertemplate="<b>%{y}</b><br>%{x:.0f} min<extra></extra>"), row=1, col=col)
    fig.update_layout(**dark("Top 10 Artists by Total Listening Time", height=440))
    grid_axes(fig)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(tickfont=dict(size=10))
    return fig


def fig_top_tracks():
    fig = make_subplots(1, 2, subplot_titles=["Person 1", "Person 2"],
                        horizontal_spacing=0.22)
    for col, (df, color) in enumerate([(p1,C1),(p2,C2)], 1):
        d = df.groupby("trackName")["minPlayed"].sum().nlargest(10).sort_values()
        labels = [t if len(t) <= 34 else t[:31]+"…" for t in d.index]
        fig.add_trace(go.Bar(
            x=d.values, y=labels, orientation="h",
            marker=dict(color=d.values, colorscale=grad(color), showscale=False,
                        line=dict(width=0)),
            text=[f" {v:.0f}" for v in d.values], textposition="outside",
            textfont=dict(color=DIM, size=10), showlegend=False,
            hovertemplate="<b>%{y}</b><br>%{x:.0f} min<extra></extra>"), row=1, col=col)
    fig.update_layout(**dark("Top 10 Tracks by Total Listening Time", height=440))
    grid_axes(fig)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(tickfont=dict(size=9))
    return fig


def fig_parallel():
    def profile(df, artists):
        rows = []
        for a in artists:
            s = df[df["artistName"]==a]
            rows.append(dict(
                artist   = a[:28],
                total    = s["minPlayed"].sum(),
                streams  = len(s),
                avg_min  = s["minPlayed"].mean(),
                skip_pct = (s["msPlayed"] < 30_000).mean() * 100,
            ))
        return pd.DataFrame(rows)

    t1 = p1.groupby("artistName")["minPlayed"].sum().nlargest(15).index
    t2 = p2.groupby("artistName")["minPlayed"].sum().nlargest(15).index
    pr1 = profile(p1, t1); pr1["who"] = 0.0
    pr2 = profile(p2, t2); pr2["who"] = 1.0
    comb = pd.concat([pr1, pr2], ignore_index=True)

    fig = go.Figure(go.Parcoords(
        line=dict(
            color=comb["who"], colorscale=[[0, C1],[1, C2]],
            showscale=True,
            colorbar=dict(tickvals=[0,1], ticktext=["Person 1","Person 2"],
                          bgcolor=CARD2, tickfont=dict(color=TEXT), outlinewidth=0)),
        dimensions=[
            dict(label="Total Min",  values=comb["total"],    range=[0, comb["total"].max()]),
            dict(label="Streams",    values=comb["streams"],  range=[0, comb["streams"].max()]),
            dict(label="Avg Min",    values=comb["avg_min"],  range=[0, comb["avg_min"].max()]),
            dict(label="Skip %",     values=comb["skip_pct"], range=[0, 100]),
        ],
        unselected=dict(line=dict(opacity=0.07, color="gray")),
    ))
    fig.update_layout(**dark("Parallel Coordinates  ·  Top-15 Artists per Person  (drag axes to filter)", height=460))
    return fig


def fig_hourly_bar():
    h1 = p1.groupby("hour")["minPlayed"].sum().reindex(range(24), fill_value=0)
    h2 = p2.groupby("hour")["minPlayed"].sum().reindex(range(24), fill_value=0)
    hrs = [f"{h:02d}:00" for h in range(24)]
    fig = go.Figure([
        go.Bar(x=hrs, y=h1.values, name="Person 1", marker_color=C1, opacity=0.85,
               hovertemplate="%{x}<br>%{y:.0f} min<extra>Person 1</extra>"),
        go.Bar(x=hrs, y=h2.values, name="Person 2", marker_color=C2, opacity=0.85,
               hovertemplate="%{x}<br>%{y:.0f} min<extra>Person 2</extra>"),
    ])
    fig.update_layout(barmode="group",
                      **dark("Hourly Listening Volume Comparison"))
    grid_axes(fig)
    fig.update_xaxes(tickfont=dict(size=9))
    return fig


def fig_animated_race():
    """Cumulative top-10 artist bar-chart race for Person 2 (11 months)."""
    months = sorted(p2["month"].unique())
    # Build union of top-15 artists (keeps y-axis stable)
    all_top = p2.groupby("artistName")["minPlayed"].sum().nlargest(15).index.tolist()
    rows = []
    for m in months:
        sub   = p2[p2["month"] <= m]
        totals = sub.groupby("artistName")["minPlayed"].sum().reindex(all_top, fill_value=0)
        top10  = totals.nlargest(10).sort_values(ascending=True)
        for artist, val in top10.items():
            rows.append({"month": m, "artist": artist[:28], "minutes": val})
    df_race = pd.DataFrame(rows)

    fig = px.bar(
        df_race, x="minutes", y="artist", animation_frame="month",
        orientation="h", range_x=[0, df_race["minutes"].max() * 1.12],
        color_discrete_sequence=[C2],
        labels={"minutes": "Total Minutes (cumulative)", "artist": ""},
    )
    fig.update_traces(
        marker=dict(line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>%{x:.0f} min<extra></extra>")
    # Speed up animation
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"]      = 700
    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 500
    # Style the play/pause button bar
    fig.update_layout(updatemenus=[dict(
        type="buttons", showactive=False, y=1.12, x=0.01, xanchor="left",
        bgcolor=C2, bordercolor="#333", font=dict(color="white"),
        buttons=[
            dict(label="▶ Play",  method="animate",
                 args=[None, {"frame":{"duration":700},"transition":{"duration":500},"fromcurrent":True}]),
            dict(label="⏸ Pause", method="animate",
                 args=[[None], {"frame":{"duration":0},"transition":{"duration":0},"mode":"immediate"}]),
        ]
    )])
    fig.update_layout(
        **dark("Person 2  ·  Cumulative Top Artists Race  (animated by month)", height=480),
        sliders=[dict(
            currentvalue=dict(prefix="Month: ", font=dict(color=TEXT)),
            font=dict(color=DIM),
            bgcolor="#1A1A1A",
            activebgcolor=C2,
            bordercolor="#333",
        )]
    )
    grid_axes(fig)
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(tickfont=dict(size=10))
    return fig


# ── Pre-build all figures ──────────────────────────────────────────────────
print("Building charts…")
FIGS = dict(
    hist       = fig_histogram(),
    line       = fig_linechart(),
    heat       = fig_heatmap(),
    box_hour   = fig_box_hour(),
    box_dow    = fig_box_dow(),
    artists    = fig_top_artists(),
    tracks     = fig_top_tracks(),
    parallel   = fig_parallel(),
    hourly_bar = fig_hourly_bar(),
    race       = fig_animated_race(),
)

# ══════════════════════════════════════════════════════════════════════════
# LAYOUT HELPERS
# ══════════════════════════════════════════════════════════════════════════
GCFG = {"displayModeBar": False, "responsive": True}

def gchart(key, extra_style=None):
    s = {"borderRadius":"12px","overflow":"hidden","marginBottom":"16px",
         "boxShadow":"0 4px 24px rgba(0,0,0,0.5)"}
    if extra_style:
        s.update(extra_style)
    return html.Div(
        dcc.Graph(figure=FIGS[key], config=GCFG),
        style=s)

def wc_card(src, label, color):
    return dbc.Col(html.Div([
        html.P(label, style={"color":color,"fontWeight":"700","fontFamily":"Inter",
                             "textAlign":"center","padding":"10px 0 6px","marginBottom":"0",
                             "fontSize":"13px","letterSpacing":"0.5px"}),
        html.Img(src=src, style={"width":"100%","display":"block"}),
    ], style={"background":CARD2,"borderRadius":"12px","overflow":"hidden",
              "marginBottom":"16px","border":f"1px solid {color}22",
              "boxShadow":"0 4px 24px rgba(0,0,0,0.5)"}), md=6)

def stat_card(label, value, icon, color, note=""):
    return dbc.Col(dbc.Card(dbc.CardBody([
        html.Div([
            html.Span(icon, style={"fontSize":"16px","marginRight":"6px"}),
            html.Span(label, style={"color":DIM,"fontSize":"9px","letterSpacing":"1.5px",
                                    "textTransform":"uppercase","fontFamily":"Inter"}),
        ], style={"display":"flex","alignItems":"center"}),
        html.H5(value, style={"color":color,"fontWeight":"700","fontFamily":"Inter",
                              "fontSize":"1.25rem","margin":"6px 0 2px","lineHeight":"1.1",
                              "overflow":"hidden","textOverflow":"ellipsis","whiteSpace":"nowrap"}),
        html.P(note, style={"color":DIM,"fontSize":"9px","margin":"0",
                            "overflow":"hidden","textOverflow":"ellipsis","whiteSpace":"nowrap"}),
    ]), style={"background":CARD2,"border":f"1px solid {color}1A",
               "borderRadius":"12px","height":"100%",
               "transition":"transform 0.2s, box-shadow 0.2s"}))

def stats_section(s, color, label, date_range):
    return [
        dbc.Row(dbc.Col(html.Div([
            html.Span("●", style={"color":color,"marginRight":"8px","fontSize":"14px"}),
            html.Span(label, style={"color":TEXT,"fontFamily":"Inter","fontWeight":"700",
                                    "fontSize":"14px"}),
            html.Span(f"  ·  {date_range}", style={"color":DIM,"fontSize":"11px","marginLeft":"8px"}),
        ]), style={"marginBottom":"8px"})),
        dbc.Row([
            stat_card("Total Streams",  s["streams"],    "🎵", color),
            stat_card("Listening Time", s["hours"],      "⏱", color),
            stat_card("Artists",        s["artists"],    "🎤", color),
            stat_card("Tracks",         s["tracks"],     "🎶", color),
            stat_card("Avg Duration",   s["avg_min"],    "📏", color),
            stat_card("Skip Rate",      s["skip_pct"],   "⏭", color, "tracks < 30 s"),
        ], className="g-2 mb-1"),
        dbc.Row([
            stat_card("Top Artist", s["top_artist"], "⭐", color),
            stat_card("Top Track",  s["top_track"],  "🏆", color, "by total time"),
        ], className="g-2 mb-3"),
    ]

# ══════════════════════════════════════════════════════════════════════════
# DASH APP
# ══════════════════════════════════════════════════════════════════════════
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap",
    ],
    title="Spotify Dashboard",
)

# Inject custom CSS for hover lift, tab styling, pulse animation
app.index_string = """<!DOCTYPE html>
<html>
<head>
  {%metas%}<title>{%title%}</title>{%favicon%}{%css%}
  <style>
    body { background-color: #0D0D0D !important; }
    * { box-sizing: border-box; }

    /* Stat card hover lift */
    .card:hover {
      transform: translateY(-3px) !important;
      box-shadow: 0 8px 32px rgba(0,0,0,0.6) !important;
    }

    /* Tab bar */
    .nav-tabs { border-bottom: 1px solid #252525 !important; }
    .nav-tabs .nav-link {
      color: #A0A0A0 !important; font-family: Inter; font-size: 12px;
      font-weight: 600; letter-spacing: 0.4px; border: none !important;
      padding: 10px 18px; border-radius: 8px 8px 0 0 !important;
      transition: color 0.2s, background 0.2s;
    }
    .nav-tabs .nav-link:hover { color: #fff !important; background: #1a1a1a !important; }
    .nav-tabs .nav-link.active {
      color: #fff !important; background: #1DB954 !important;
      border-bottom: none !important;
    }
    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0D0D0D; }
    ::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }

    /* Pulse animation on header dot */
    @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.35;} }
    .pulse { animation: pulse 2.4s ease-in-out infinite; }

    /* Fade-in for tab content */
    .tab-content { animation: fadeIn 0.35s ease; }
    @keyframes fadeIn { from{opacity:0;transform:translateY(6px);} to{opacity:1;transform:none;} }
  </style>
</head>
<body>{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>"""

tab_style = {"padding": "20px 0 0"}

app.layout = dbc.Container([

    # ── Header ───────────────────────────────────────────────────────────
    dbc.Row(dbc.Col(html.Div([
        html.Span("●", className="pulse",
                  style={"color":C1,"fontSize":"22px","marginRight":"10px",
                         "verticalAlign":"middle"}),
        html.Span("Spotify Streaming Dashboard",
                  style={"color":TEXT,"fontFamily":"Inter","fontWeight":"700",
                         "fontSize":"22px","verticalAlign":"middle"}),
        html.Span("  Person 1 (2025)  vs  Person 2 (2020–21)",
                  style={"color":DIM,"fontFamily":"Inter","fontSize":"12px",
                         "marginLeft":"14px","verticalAlign":"middle"}),
    ], style={"padding":"22px 0 14px",
              "borderBottom":f"1px solid #252525",
              "marginBottom":"20px"}))),

    # ── Stats ─────────────────────────────────────────────────────────────
    *stats_section(S1, C1, "Person 1", S1["date_range"]),
    *stats_section(S2, C2, "Person 2", S2["date_range"]),

    html.Hr(style={"borderColor":"#1E1E1E","margin":"8px 0 20px"}),

    # ── Tabs ──────────────────────────────────────────────────────────────
    dbc.Tabs(active_tab="tab-dist", children=[

        dbc.Tab(tab_id="tab-dist", label="📊 Distribution", children=[
            html.Div([gchart("hist")], style=tab_style),
        ]),

        dbc.Tab(tab_id="tab-line", label="📈 Timeline", children=[
            html.Div([gchart("line")], style=tab_style),
        ]),

        dbc.Tab(tab_id="tab-heat", label="🔥 Heatmap", children=[
            html.Div([gchart("heat")], style=tab_style),
        ]),

        dbc.Tab(tab_id="tab-box", label="📦 Box Plots", children=[
            html.Div([
                gchart("box_hour"),
                gchart("box_dow"),
            ], style=tab_style),
        ]),

        dbc.Tab(tab_id="tab-artists", label="🎤 Artists", children=[
            html.Div([
                gchart("artists"),
                dbc.Row([
                    wc_card(WC["a1"], "Person 1  ·  Artist Word Cloud", C1),
                    wc_card(WC["a2"], "Person 2  ·  Artist Word Cloud", C2),
                ]),
            ], style=tab_style),
        ]),

        dbc.Tab(tab_id="tab-tracks", label="🎵 Tracks", children=[
            html.Div([
                gchart("tracks"),
                dbc.Row([
                    wc_card(WC["t1"], "Person 1  ·  Track Word Cloud", C1),
                    wc_card(WC["t2"], "Person 2  ·  Track Word Cloud", C2),
                ]),
            ], style=tab_style),
        ]),

        dbc.Tab(tab_id="tab-deep", label="⚡ Deep Dive", children=[
            html.Div([
                gchart("parallel"),
                gchart("hourly_bar"),
                gchart("race"),
            ], style=tab_style),
        ]),

    ]),

    html.Div(style={"height":"40px"}),

], fluid=True, style={"backgroundColor":BG, "minHeight":"100vh",
                      "paddingLeft":"28px","paddingRight":"28px"})


if __name__ == "__main__":
    print("Dashboard ready  ->  http://127.0.0.1:8050")
    app.run(debug=False, port=8050)
