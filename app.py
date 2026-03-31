import dash
import dash_bootstrap_components as dbc

from layout import layout

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap",
    ],
    title="Spotify Dashboard",
)

app.layout = layout

if __name__ == "__main__":
    print("Dashboard ready  ->  http://127.0.0.1:8050")
    app.run(debug=False, port=8050)
