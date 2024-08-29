import os
import time
import dash
from dash import html, dcc, Input, Output, State
from pathlib import Path
import yaml
import dash_bootstrap_components as dbc
import urllib.parse
import subprocess, sys

app = dash.get_app()

dash.register_page(__name__,  path_template="/config/<config_name>")
experiment_configs_path = Path(os.path.join("..", "config", "experiments"))


def decompose(configurations, recurse=False, level=0):

    configurations_html = []
    try:
        for k, v in configurations.items():
            if type(v) is not dict:
                configurations_html.append(
                    dbc.Row([
                        html.Div(className="modalEditor", children=[
                            html.Div(className="modal-content", children=[
                                html.Div(className="modal_configuration_parameters", children=[
                                    dbc.Row([dbc.Col(dbc.Button(className="bi bi-plus rounded-circle modalEndwagon", outline=True, color="primary"), width="auto"),], style={"margin-left": 0}, className="modalEndwagon"),
                                ]),
                                dbc.Row([dbc.Col(dbc.Button("Save", color="primary", className="modal_save_btn me-1", n_clicks=0), width="auto"),dbc.Col(dbc.Button("Cancel", id="modal_close", color="primary", className="me-1 close", n_clicks=0), width="auto"),], style={"margin-top":30})
                            ]),
                        ]),

                        dbc.Col(dbc.Button(className="bi bi-plus rounded-circle", outline=True, color="primary"), width="auto"),
                        dbc.Col([dbc.Input(className="keyfield", value=k, type="text", list="parameters"),
                                 html.Span(className="tooltiptext")], className="tooltip keyfield_col"),
                        dbc.Col(dbc.Input(className="valuefield", value=v, type="text"), className="valuefield_col"),
                        dbc.Col(dbc.Button(className="bi bi-pencil", outline=True, color="secondary"), width="auto") if "override /" in k else dbc.Col(width="auto"),
                        dbc.Col(dbc.Button(className="bi bi-trash", outline=True, color="secondary")),
                    ], style={"margin-left": 100*level}, className="child")
                )
            else:

                children = decompose(v, True, level=level + 1)

                configurations_html.append(

                dbc.Row([
                    dbc.Row([
                        dbc.Col(dbc.Button(className="bi bi-plus rounded-circle", outline=True, color="primary"), width="auto"),
                        dbc.Col([dbc.Input(className="keyfield", value=k, type="text", list="parameters"),
                                 html.Span(className="tooltiptext")], className="tooltip keyfield_col"),
                        dbc.Col(dbc.Button(className="bi bi-trash", outline=True, color="secondary")),
                    ], style={"margin-left": 100*level}, className="parent"),
                      *children
                    ], className="uniter"
                    )

                )

        return configurations_html
    except Exception as e:
        print(e, k, v)


def simplify_dict(in_dict):
    new_dict = {}

    for k, v in in_dict.items():
        if type(v) is list:
            intermediate_dict = {}
            for element in v:
                if type(element) is dict:
                    for key, value in element.items():
                        intermediate_dict[key] = value
                    intermediate_dict = simplify_dict(intermediate_dict)
                    new_dict[k] = intermediate_dict
                else:
                    if k not in new_dict.keys():
                        new_dict[k] = [element]
                    else:
                        new_dict[k].append(element)
        else:
            new_dict[k] = v
    return new_dict


def layout(config_name=None):

    if config_name is None:
        return None

    config_name = urllib.parse.unquote(config_name)
    exp_path = experiment_configs_path / config_name

    try:
        env_key = "UI_TITLE"
        title = os.environ[env_key]
    except Exception as e:
        print(f"No ui title, using default")
        title = "Experiment Configurator"

    html_components = [dbc.Row([dbc.Col([html.H4(title, style={"height": 40})]),
                                dbc.Col([html.Div(html.Img(src=dash.get_asset_url('_innofw_.svg'), style={"height": 40, "width": 60}),
                                   className="self-align-right")]),html.Span(className="border-bottom")],style={"margin-top": 10, "margin-bottom": 10, "margin-right": 15}),
                       html.Br(),
                       dbc.Input(id="config_name", value=config_name, type="text", style={"width": "auto"})]
    if exp_path.exists():
        with open(exp_path, "r") as yamlfile:
            data = yaml.load(yamlfile, Loader=yaml.FullLoader)
            yamlfile.close()

        if data:
            data = simplify_dict(data)
            configurations_html = decompose(data)
            html_components.extend(configurations_html)
    else:
        with open(experiment_configs_path / "empty_template.yaml", "r") as yamlfile:
            data = yaml.load(yamlfile, Loader=yaml.FullLoader)
            yamlfile.close()

        if data:
            data = simplify_dict(data)
            configurations_html = decompose(data)
            html_components.extend(configurations_html)


    html_components.append(dbc.Row([
                        dbc.Col(dbc.Button(className="bi bi-plus rounded-circle endwagon", outline=True, color="primary"), width="auto"),
                    ], style={"margin-left": 0}, className="endwagon", id="endwagon"))

    html_components.append(html.Br())
    html_components.append(html.Div(id='analytics-output',
                                    children=[
                                        dash.html.Datalist(children=[
                                              dash.html.Option("batch_size"),
                                              dash.html.Option("epochs"),
                                            dash.html.Option("learning_rate"),
                                            dash.html.Option("random_seed"),
                                            dash.html.Option("task"),
                                            dash.html.Option("weights_freq"),
                                            dash.html.Option("project"),
                                            dash.html.Option("defaults"),
                                            dash.html.Option("accelerator"),
                                            dash.html.Option("gpus"),
                                            dash.html.Option("devices"),
                                            dash.html.Option("device"),
                                            dash.html.Option("ckpt_path"),
                                                ],
                                                    id="parameters")]))

    html_components.append(html.Progress(id="progress_id", style={"visibility": "hidden"}))
    html_components.append(html.Div(id='infer-analytics-output'))

    html_components.append(dbc.Row([
        dbc.Col(
            dcc.Link(dbc.Button("Back", id="back_btn", color="secondary", className="me-1", n_clicks=0),
            href="/", refresh=True), width="auto"),


        dbc.Col(dbc.Button("Save", id="save_btn", color="primary", className="save_btn me-1", n_clicks=0), width="auto"),

        dbc.Col(dbc.Button("Delete", id="delete_btn", color="danger", className="me-1", n_clicks=0), width="auto"),
        dbc.Col(dbc.Button("Duplicate", id="duplicate_btn", color="info", className="me-1 duplicate_btn", n_clicks=0)),

        dbc.Col(dbc.Button("Start Training", id="strain_btn", color="success", className="me-2", n_clicks=0)),
        dbc.Col(dbc.Button("Start Inference", id="sinfer_btn", color="dark", className="me-2", n_clicks=0)),

        dbc.Row(html.Div(className="modal-content", children=[
             dbc.ModalHeader(dbc.ModalTitle("Saving")),
             dbc.ModalBody(f"Configuration {config_name} is saved"),
             dbc.ModalFooter(
                 [
                    dbc.Button("Ok", id="confirm_save_btn", className="ms-auto", n_clicks=0)
                 ],),
             ]),
            id="confirm_save_modal",
            className="modalSavingConfirmator",
            ),






        dbc.Modal(
            [dbc.ModalHeader(dbc.ModalTitle("Deletion")),
            dbc.ModalBody(f"Are you sure you want to delete {config_name}?"),
            dbc.ModalFooter(
                    [
                        dcc.Link(dbc.Button("Delete", id="confirm_delete_btn", color="danger", className="me-1", n_clicks=0),
                                     href=f"/delete_config/{config_name}", refresh=True),
                        dbc.Button(
                            "Cancel", id="cancel", className="ms-auto", n_clicks=0)
                    ]),],
            id="modal",
            is_open=False),
    ]))

    config_edition_page_layout = html.Div(children=html_components, className="configuration_parameters", id="configuration_parameters",  style={"margin-left": 20})

    return config_edition_page_layout

#
# @app.callback(
#     Output("confirm_save_modal", "is_open"),
#     [Input("save_btn", "n_clicks"), Input("confirm_save_btn", "n_clicks")],
#     State("confirm_save_modal", "is_open"),
#     prevent_initial_call=True
# )
# def toggle_confirm_modal(n1, n2, is_open):
#     if n2:
#         return False

@app.callback(
    Output("modal", "is_open"),
    [Input("delete_btn", "n_clicks"), Input("confirm_delete_btn", "n_clicks"),  Input("cancel", "n_clicks")],
    State("modal", "is_open"),
    prevent_initial_call=True
)
def toggle_modal(n1, n2, n3, is_open):
    if n1 or n3:
        return not is_open


@app.long_callback(
    Output("analytics-output", "children"),
    Input("strain_btn", "n_clicks"),
    Input("config_name", "value"),

    progress=[Output("analytics-output", "children")],
    prevent_initial_call=True
)
def on_strain_btn_button_click(set_progress, n, config_name):

    if n > 0:
        run_env = os.environ.copy()
        run_env["NO_CLI"] = "True"
        run_env["PYTHONPATH"] = ".."

        cmd = [sys.executable, "../train.py", f"experiments={config_name}"]

        text_output = []
        err_output = []
        out = html.Div(id="process_output")

        with open("my_command.out", "w") as subprocess_out:
            with open("my_command.err", "w") as subprocess_err:
                with subprocess.Popen(cmd, stdout=subprocess_out, stderr=subprocess_err, env=run_env) as process:
                    with open("my_command.out", "r") as subprocess_outin:
                        with open("my_command.err", "r") as subprocess_errin:

                            while True:
                                out_text = subprocess_outin.read()
                                out_err = subprocess_errin.read()

                                if out_err:
                                    err_output.append(html.P(out_err))

                                if not out_text and process.poll() is None:
                                    time.sleep(0.5)
                                    continue

                                text_output.append(html.P(out_text))

                                out = dbc.Row(id="process_output", children=[dbc.Col(text_output), dbc.Col(err_output)])
                                set_progress(out)
                                if out_text == '' and process.poll() != None:
                                    break
        return out


@app.long_callback(
    Output("infer-analytics-output", "children"),
    Input("sinfer_btn", "n_clicks"),
    Input("config_name", "value"),

    progress=[Output("infer-analytics-output", "children")],
    prevent_initial_call=True
)
def on_sinfer_btn_button_click(set_progress, n, config_name):

    if n > 0:
        run_env = os.environ.copy()
        run_env["NO_CLI"] = "True"
        run_env["PYTHONPATH"] = ".."

        cmd = [sys.executable, "../infer.py", f"experiments={config_name}"]

        text_output = []
        err_output = []
        out = html.Div(id="infer_process_output")

        with open("infer.out", "w") as subprocess_out:
            with open("infer.err", "w") as subprocess_err:
                with subprocess.Popen(cmd, stdout=subprocess_out, stderr=subprocess_err, env=run_env) as process:
                    with open("infer.out", "r") as subprocess_outin:
                        with open("infer.err", "r") as subprocess_errin:

                            while True:
                                out_text = subprocess_outin.read()
                                out_err = subprocess_errin.read()

                                if out_err:
                                    err_output.append(html.P(out_err))

                                if not out_text and process.poll() is None:
                                    time.sleep(0.5)
                                    continue

                                text_output.append(html.P(out_text))

                                out = dbc.Row(id="infer_process_output", children=[dbc.Col(text_output), dbc.Col(err_output)])
                                set_progress(out)
                                if out_text == '' and process.poll() != None:
                                    break
        return out

