from dash import dcc, html
import dash_bootstrap_components as dbc

from config import C1, C2, BG, CARD2, DIM, TEXT
from data import S1, S2, WC
from charts import FIGS

GCFG      = {"displayModeBar": False, "responsive": True}
tab_style = {"padding": "20px 0 0"}


def gchart(key, extra_style=None):
    s = {"borderRadius": "12px", "overflow": "hidden", "marginBottom": "16px",
         "boxShadow": "0 4px 24px rgba(0,0,0,0.5)"}
    if extra_style:
        s.update(extra_style)
    return html.Div(dcc.Graph(figure=FIGS[key], config=GCFG), style=s)


def wc_card(src, label, color):
    return dbc.Col(html.Div([
        html.P(label, style={"color": color, "fontWeight": "700", "fontFamily": "Inter",
                             "textAlign": "center", "padding": "10px 0 6px", "marginBottom": "0",
                             "fontSize": "13px", "letterSpacing": "0.5px"}),
        html.Img(src=src, style={"width": "100%", "display": "block"}),
    ], style={"background": CARD2, "borderRadius": "12px", "overflow": "hidden",
              "marginBottom": "16px", "border": f"1px solid {color}22",
              "boxShadow": "0 4px 24px rgba(0,0,0,0.5)"}), md=6)


def stat_card(label, value, icon, color, note=""):
    return dbc.Col(dbc.Card(dbc.CardBody([
        html.Div([
            html.Span(icon,  style={"fontSize": "16px", "marginRight": "6px"}),
            html.Span(label, style={"color": DIM, "fontSize": "9px", "letterSpacing": "1.5px",
                                    "textTransform": "uppercase", "fontFamily": "Inter"}),
        ], style={"display": "flex", "alignItems": "center"}),
        html.H5(value, style={"color": color, "fontWeight": "700", "fontFamily": "Inter",
                              "fontSize": "1.25rem", "margin": "6px 0 2px", "lineHeight": "1.1",
                              "overflow": "hidden", "textOverflow": "ellipsis", "whiteSpace": "nowrap"}),
        html.P(note, style={"color": DIM, "fontSize": "9px", "margin": "0",
                            "overflow": "hidden", "textOverflow": "ellipsis", "whiteSpace": "nowrap"}),
    ]), style={"background": CARD2, "border": f"1px solid {color}1A",
               "borderRadius": "12px", "height": "100%",
               "transition": "transform 0.2s, box-shadow 0.2s"}))


def stats_section(s, color, label, date_range):
    return [
        dbc.Row(dbc.Col(html.Div([
            html.Span("●", style={"color": color, "marginRight": "8px", "fontSize": "14px"}),
            html.Span(label, style={"color": TEXT, "fontFamily": "Inter", "fontWeight": "700",
                                    "fontSize": "14px"}),
            html.Span(f"  ·  {date_range}", style={"color": DIM, "fontSize": "11px", "marginLeft": "8px"}),
        ]), style={"marginBottom": "8px"})),
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


layout = dbc.Container([

    # Header
    dbc.Row(dbc.Col(html.Div([
        html.Span("●", className="pulse",
                  style={"color": C1, "fontSize": "22px", "marginRight": "10px",
                         "verticalAlign": "middle"}),
        html.Span("Spotify Streaming Dashboard",
                  style={"color": TEXT, "fontFamily": "Inter", "fontWeight": "700",
                         "fontSize": "22px", "verticalAlign": "middle"}),
        html.Span("  Person 1 (2025)  vs  Person 2 (2020–21)",
                  style={"color": DIM, "fontFamily": "Inter", "fontSize": "12px",
                         "marginLeft": "14px", "verticalAlign": "middle"}),
    ], style={"padding": "22px 0 14px",
              "borderBottom": "1px solid #252525",
              "marginBottom": "20px"}))),

    # Stats
    *stats_section(S1, C1, "Person 1", S1["date_range"]),
    *stats_section(S2, C2, "Person 2", S2["date_range"]),

    html.Hr(style={"borderColor": "#1E1E1E", "margin": "8px 0 20px"}),

    # Tabs
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
            html.Div([gchart("box_hour"), gchart("box_dow")], style=tab_style),
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
            html.Div([gchart("parallel"), gchart("hourly_bar"), gchart("race")], style=tab_style),
        ]),

    ]),

    html.Div(style={"height": "40px"}),

], fluid=True, style={"backgroundColor": BG, "minHeight": "100vh",
                      "paddingLeft": "28px", "paddingRight": "28px"})
