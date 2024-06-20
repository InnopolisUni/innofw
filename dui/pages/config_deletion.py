import os
import dash
from dash import html, dcc, callback, Input, Output, State
from pathlib import Path
import dash_bootstrap_components as dbc
import urllib.parse


def layout(config_name=None):

    if config_name is None:
        return None

    config_name = urllib.parse.unquote(config_name)
    print(f"delete {config_name}")
    experiment_configs_path = Path(os.path.join("..", "config", "experiments"))
    os.remove(experiment_configs_path / config_name)
    return dcc.Location(pathname="/", id="Sdfa")


dash.register_page(__name__,  path_template="/delete_config/<config_name>")

