import dash
import dash_bootstrap_components as dbc
from dash import html

from dash.long_callback import DiskcacheLongCallbackManager
import diskcache

cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

app = dash.Dash(__name__,
                long_callback_manager=long_callback_manager,
                use_pages=True,
                external_stylesheets=["bootstrap.min.css", ("icons/", "font/bootstrap-icons.css"), "dui.css"],
                external_scripts=[
				  "jquery-3.3.1.slim.min.js",
                                    "popper.min.js",
                                    "bootstrap.min.js",
				  "dui.js",])  # initialising dash app
dash.register_page(__name__,  path_template="/")


def layout():
    lout = dbc.Container([

        dbc.Row([dbc.Col([html.H4("Experiment Configurator", style={"height": 40})]),
                 dbc.Col([html.Div(html.Img(src=dash.get_asset_url('_innofw_.svg'), style={"height": 40, "width": 60}),
                                   className="self-align-right")]),
                 html.Span(className="border-bottom")],

                style={"margin-top": 10, "margin-bottom": 10, "margin-right": 5}),

        dash.page_container])
    return lout


if __name__ == '__main__':
    app.run_server()
