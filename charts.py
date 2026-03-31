import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config import C1, C2, DOW, DIM, CARD2, TEXT, rgba, grad, dark, grid_axes
from data import p1, p2


def fig_histogram():
    fig = make_subplots(1, 2, subplot_titles=["Person 1  (2025)", "Person 2  (2020–21)"],
                        horizontal_spacing=0.08)
    for col, (df, color) in enumerate([(p1, C1), (p2, C2)], 1):
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
    for row, (df, color) in enumerate([(p1, C1), (p2, C2)], 1):
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
    for col, (df, cmap) in enumerate([(p1, "Greens"), (p2, "Reds")], 1):
        heat = (df.groupby(["dayOfWeek", "hour"])["minPlayed"]
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
    for col, (df, color) in enumerate([(p1, C1), (p2, C2)], 1):
        for h in range(24):
            vals = df.loc[df["hour"] == h, "minPlayed"].clip(upper=10).values
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
    for col, (df, color) in enumerate([(p1, C1), (p2, C2)], 1):
        for d in DOW:
            vals = df.loc[df["dayOfWeek"] == d, "minPlayed"].clip(upper=10).values
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
    for col, (df, color) in enumerate([(p1, C1), (p2, C2)], 1):
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
    for col, (df, color) in enumerate([(p1, C1), (p2, C2)], 1):
        d = df.groupby("trackName")["minPlayed"].sum().nlargest(10).sort_values()
        labels = [t if len(t) <= 34 else t[:31] + "…" for t in d.index]
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
            s = df[df["artistName"] == a]
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
            color=comb["who"], colorscale=[[0, C1], [1, C2]],
            showscale=True,
            colorbar=dict(tickvals=[0, 1], ticktext=["Person 1", "Person 2"],
                          bgcolor=CARD2, tickfont=dict(color=TEXT), outlinewidth=0)),
        dimensions=[
            dict(label="Total Min", values=comb["total"],    range=[0, comb["total"].max()]),
            dict(label="Streams",   values=comb["streams"],  range=[0, comb["streams"].max()]),
            dict(label="Avg Min",   values=comb["avg_min"],  range=[0, comb["avg_min"].max()]),
            dict(label="Skip %",    values=comb["skip_pct"], range=[0, 100]),
        ],
        unselected=dict(line=dict(opacity=0.07, color="gray")),
    ))
    fig.update_layout(**dark("Parallel Coordinates  ·  Top-15 Artists per Person  (drag axes to filter)", height=460))
    return fig


def fig_hourly_bar():
    h1  = p1.groupby("hour")["minPlayed"].sum().reindex(range(24), fill_value=0)
    h2  = p2.groupby("hour")["minPlayed"].sum().reindex(range(24), fill_value=0)
    hrs = [f"{h:02d}:00" for h in range(24)]
    fig = go.Figure([
        go.Bar(x=hrs, y=h1.values, name="Person 1", marker_color=C1, opacity=0.85,
               hovertemplate="%{x}<br>%{y:.0f} min<extra>Person 1</extra>"),
        go.Bar(x=hrs, y=h2.values, name="Person 2", marker_color=C2, opacity=0.85,
               hovertemplate="%{x}<br>%{y:.0f} min<extra>Person 2</extra>"),
    ])
    fig.update_layout(barmode="group", **dark("Hourly Listening Volume Comparison"))
    grid_axes(fig)
    fig.update_xaxes(tickfont=dict(size=9))
    return fig


def fig_animated_race():
    months  = sorted(p2["month"].unique())
    all_top = p2.groupby("artistName")["minPlayed"].sum().nlargest(15).index.tolist()
    rows = []
    for m in months:
        sub    = p2[p2["month"] <= m]
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
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"]      = 700
    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 500
    fig.update_layout(updatemenus=[dict(
        type="buttons", showactive=False, y=1.12, x=0.01, xanchor="left",
        bgcolor=C2, bordercolor="#333", font=dict(color="white"),
        buttons=[
            dict(label="▶ Play",  method="animate",
                 args=[None, {"frame": {"duration": 700}, "transition": {"duration": 500}, "fromcurrent": True}]),
            dict(label="⏸ Pause", method="animate",
                 args=[[None], {"frame": {"duration": 0}, "transition": {"duration": 0}, "mode": "immediate"}]),
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
