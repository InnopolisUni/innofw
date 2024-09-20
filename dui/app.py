import dash
import dash_bootstrap_components as dbc
from dash import html

from dash.long_callback import DiskcacheLongCallbackManager
import diskcache
import os



try:
    env_key = "UI_TITLE"
    title = os.environ[env_key]
except Exception as e:
    print(f"No ui title, using default")
    title = "Experiment Configurator"

layout = dbc.Container([
        dbc.Row([dbc.Col([html.H4(title, style={"height": 40})]),
                 html.Span(className="border-bottom")],
                style={"margin-top": 10, "margin-bottom": 10, "margin-right": 5}),
        dash.page_container])

if __name__ == '__main__':
    cache = diskcache.Cache("./cache")
    long_callback_manager = DiskcacheLongCallbackManager(cache)

    app = dash.Dash(__name__,
                    long_callback_manager=long_callback_manager,
                    use_pages=True,
                    external_stylesheets=["bootstrap.min.css",
                                          ("icons/", "font/bootstrap-icons.css"),
                                          "dui.css",],
                    external_scripts=["jquery-3.3.1.slim.min.js",
                                      "popper.min.js",
                                      "bootstrap.min.js",
                                      "dui.js"])  # initialising dash app

    dash.register_page("inintial", path_template="/", layout=layout)
    app.run_server(host='0.0.0.0')
