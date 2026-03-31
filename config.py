DOW = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

C1, C2 = "#1DB954", "#FF6B6B"
BG     = "#0D0D0D"
CARD   = "#161616"
CARD2  = "#1E1E1E"
TEXT   = "#FFFFFF"
DIM    = "#A0A0A0"


def hex_rgb(h):
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgba(h, a):
    r, g, b = hex_rgb(h)
    return f"rgba({r},{g},{b},{a})"


def grad(color):
    r, g, b = hex_rgb(color)
    return [[0, f"rgba({r},{g},{b},0.25)"], [1, color]]


def dark(title="", height=420, legend=True):
    d = dict(
        title      = dict(text=title, font=dict(color=TEXT, size=14, family="Inter"),
                          x=0.01, xanchor="left"),
        height     = height,
        paper_bgcolor = CARD2,
        plot_bgcolor  = "#111111",
        font       = dict(color=DIM, family="Inter, sans-serif", size=11),
        margin     = dict(l=14, r=14, t=48, b=14),
        legend     = dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT)) if legend else None,
        transition = dict(duration=500, easing="cubic-in-out"),
        hoverlabel = dict(bgcolor="#1E1E1E", font_color=TEXT, bordercolor="#333"),
    )
    if not legend:
        d.pop("legend")
    return d


def grid_axes(fig):
    fig.update_xaxes(gridcolor="#252525", zerolinecolor="#333", linecolor="#333")
    fig.update_yaxes(gridcolor="#252525", zerolinecolor="#333", linecolor="#333")
