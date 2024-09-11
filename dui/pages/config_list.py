import os
import dash
from pathlib import Path
import dash_bootstrap_components as dbc

from dash import dash_table, dcc, html

import pandas as pd

dash.register_page(
    __name__,
    path='/config_list',
    title='config_list'
)

experiment_configs_path = Path(os.path.join("..", "config", "experiments"))

def layout():
    experiment_configs = []
    for p in experiment_configs_path.rglob('*'):
        if str(p).endswith(".yaml"):
            experiment_configs.append(str(p.relative_to(experiment_configs_path)))

    df = pd.DataFrame().from_dict({"Experiments": experiment_configs})

    df['id'] = list(range(len(df)))
    df.set_index('id', inplace=True, drop=False)

    for i in range(0, 9999999999):
        exp_name = f"New Experiment{i}.yaml" if i > 0 else "New Experiment.yaml"
        p = experiment_configs_path / exp_name
        if not p.exists():
            break

    try:
        env_key = "UI_TITLE"
        title = os.environ[env_key]
    except Exception as e:
        print(f"No ui title, using default")
        title = "Experiment Configurator"

    config_list = dbc.Container([

        dbc.Row([dbc.Col([html.H4(title, style={"height": 40})]),
                 dbc.Col([html.Div(html.Img(src=dash.get_asset_url('_innofw_.svg'), style={"height": 40, "width": 60}),
                                   className="self-align-right")]),
                 html.Span(className="border-bottom")],

                style={"margin-top": 10, "margin-bottom": 10}),


        dbc.Row([
        html.Div(id="parent", style={"width": 300}, children=[
            dcc.Link(dbc.Button('Create new experiment', id='create_exp', style={"width": 300, "margin-bottom": 5}, n_clicks=0),
                     href=f"/config/{exp_name}", refresh=True),

            dash_table.DataTable(
                id='table',
                columns=[
                    {"name": i, "id": i, "deletable": False, "selectable": False} for i in ["Experiments"]
                ],
                data=df.to_dict('records'),
                editable=False,
                filter_action="native",
                filter_options={"placeholder_text": "Search"},
                sort_action='none',
                sort_mode="multi",
                column_selectable="single",
                row_selectable=False,
                row_deletable=False,
                selected_columns=[],
                selected_rows=[],
                page_action="native",
                page_current=0,
                page_size=20,
                style_header={'backgroundColor': 'white',
                              'fontWeight': 'bold',
                              'textAlign': 'left'
                              },
                style_cell={'textAlign': 'left'}
            ),
        ]),])

    ])

    return config_list