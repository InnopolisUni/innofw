import os
import dash
from dash import html, dcc, callback, Input, Output, State
from pathlib import Path
import dash_bootstrap_components as dbc
import urllib.parse

app = dash.get_app()

dash.register_page(__name__,  path_template="/delete_config/<config_name>")
experiment_configs_path = Path(os.path.join("..", "config", "experiments"))


def layout(config_name=None):
    config_name = urllib.parse.unquote(config_name)
    print(f"delete {config_name}")
    os.remove(experiment_configs_path / config_name)
    return dcc.Location(pathname="/", id="Sdfa")


