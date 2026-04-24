import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, html
import plotly.graph_objects as go

from layout import layout
from config import C1, C2, CARD2, DIM, TEXT, grad, dark, grid_axes
from recommender import RECS

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap",
    ],
    title="Spotify Dashboard",
)

app.layout = layout


@app.callback(
    Output("reco-chart",   "figure"),
    Output("reco-cm",      "figure"),
    Output("reco-summary", "children"),
    Input("reco-person",   "value"),
    Input("reco-model",    "value"),
)
def update_reco(person, model):
    df, acc, cm   = RECS[(person, model)]
    color         = C1 if person == 'p1' else C2
    person_label  = "Person 1" if person == 'p1' else "Person 2"
    source_label  = "Person 2" if person == 'p1' else "Person 1"
    model_label   = "Decision Tree" if model == 'dt' else "KNN (k=5)"

    # ── Recommendation bar chart ─────────────────────────────────────────────
    top = df.head(20)

    if top.empty:
        fig = go.Figure()
        fig.update_layout(**dark("No tracks passed the model threshold — try the other model", height=200))
    else:
        y_labels = [
            f"{r['track'][:36]}  —  {r['artist'][:24]}"
            for _, r in top.iterrows()
        ]
        fig = go.Figure(go.Bar(
            x=top['confidence'].values,
            y=y_labels,
            orientation='h',
            marker=dict(
                color=top['confidence'].values,
                colorscale=grad(color),
                showscale=False,
                line=dict(width=0),
            ),
            text=[f"  {c:.0%}" for c in top['confidence']],
            textposition='outside',
            textfont=dict(color=DIM, size=9),
            hovertemplate='<b>%{y}</b><br>Confidence: %{x:.1%}<extra></extra>',
        ))
        fig.update_layout(**dark(
            f'{person_label}  ·  Top Tracks from {source_label}\'s Library  [{model_label}]',
            height=max(400, len(top) * 30 + 100),
        ))
        fig.update_xaxes(range=[0, 1.18], title_text="Model Confidence",
                         showgrid=True, gridcolor='#252525')
        fig.update_yaxes(tickfont=dict(size=9))
        grid_axes(fig)

    # ── Confusion matrix heatmap ─────────────────────────────────────────────
    tn, fp, fn, tp_val = cm.ravel()
    total = cm.sum()

    labels    = ["Not Liked", "Liked"]
    z_vals    = [[tn, fp], [fn, tp_val]]
    z_pct     = [[tn / total, fp / total], [fn / total, tp_val / total]]

    annotations = []
    for i, row in enumerate(z_vals):
        for j, val in enumerate(row):
            cell_label = {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}[(i, j)]
            annotations.append(dict(
                x=j, y=i,
                text=f"<b>{val}</b><br>{cell_label}<br>{z_pct[i][j]:.1%}",
                showarrow=False,
                font=dict(color=TEXT, size=13, family="Inter"),
            ))

    cm_fig = go.Figure(go.Heatmap(
        z=z_vals,
        x=labels,
        y=labels,
        colorscale=[[0, "#111111"], [1, color]],
        showscale=False,
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))
    cm_fig.update_layout(
        annotations=annotations,
        xaxis=dict(title="Predicted", tickfont=dict(color=TEXT), side="bottom",
                   gridcolor="#252525", linecolor="#333"),
        yaxis=dict(title="Actual", tickfont=dict(color=TEXT), autorange="reversed",
                   gridcolor="#252525", linecolor="#333"),
        **dark(f"Confusion Matrix  ·  {person_label}  [{model_label}]  —  Accuracy: {acc:.1%}",
               height=340),
    )

    # ── Summary stat cards ───────────────────────────────────────────────────
    precision = tp_val / (tp_val + fp) if (tp_val + fp) > 0 else 0
    recall    = tp_val / (tp_val + fn) if (tp_val + fn) > 0 else 0

    def _stat(label, value, note=""):
        return dbc.Col(dbc.Card(dbc.CardBody([
            html.Div(label, style={"color": DIM, "fontSize": "10px",
                                   "letterSpacing": "1px", "textTransform": "uppercase",
                                   "fontFamily": "Inter"}),
            html.H4(value, style={"color": color, "fontWeight": "700",
                                  "margin": "6px 0 2px", "fontFamily": "Inter"}),
            html.Div(note, style={"color": DIM, "fontSize": "10px", "fontFamily": "Inter"}),
        ]), style={"background": CARD2, "borderRadius": "12px",
                   "border": f"1px solid {color}22",
                   "boxShadow": "0 4px 24px rgba(0,0,0,0.4)"}), md=2)

    summary = dbc.Row([
        _stat("Model",       model_label),
        _stat("Accuracy",    f"{acc:.1%}",         "held-out test set"),
        _stat("Precision",   f"{precision:.1%}",   "of predicted liked, truly liked"),
        _stat("Recall",      f"{recall:.1%}",      "of truly liked, correctly found"),
        _stat("Recommended", str(len(df)),          f"tracks from {source_label}"),
        _stat("Showing",     f"Top {min(20,len(df))}", "sorted by confidence"),
    ], className="g-2 mb-3")

    return fig, cm_fig, summary


if __name__ == "__main__":
    print("Dashboard ready  ->  http://127.0.0.1:8050")
    app.run(debug=False, port=8050)
