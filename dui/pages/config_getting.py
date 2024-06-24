import os
import dash
from dash import html, dcc, Input, Output, State
from pathlib import Path
import dash_bootstrap_components as dbc
from flask import request, Flask
app = Flask(__name__)
import yaml
import json

configs_path = Path(os.path.join("..", "config"))
flaskapp = dash.get_app().server

# DO NOT DELETE THIS COMMENTED PIECE OF CODE IT IS NEEDED, BECAUSE WITHOUT IT THIS PAGE IS NOT LOADED ON START
# AND FLASK WON'T LISTEN TO THE URL
# dash.register_page(__name__,  path_template="/None")



@flaskapp.route("/get_config", methods = ["GET"])
def get_config():
    request_body = request.args.get('config_name')
    request_body += ".yaml" if not request_body.endswith(".yaml") else ""
    exp_path = configs_path / request_body
    data = ""
    if exp_path.exists():
        with open(exp_path, "r") as yamlfile:
            try:
                data = yaml.load(yamlfile, Loader=yaml.FullLoader)

                if data:
                    data = simplify_dict(data)
            except yaml.YAMLError as exc:
                print("ERROR:", exc)
    return json.dumps({'success': True, 'configuration_parameters': data}), 200, {'ContentType':'application/json'}


def parse_config_from_html(config_parameters, recurse=False):
    config_dict = {}
    parent = ""
    for el in config_parameters:

        if el["props"]["className"] == "uniter":
            # recurse
            d = parse_config_from_html(el["props"]["children"], recurse=True)
            config_dict.update(d)
        elif el["props"]["className"] == "parent":
            k = el["props"]["children"][1]["props"]["children"]["props"]["value"]
            v = el["props"]["children"][2]["props"]["children"]["props"]["value"] if len( el["props"]["children"])==4 else None
            parent = k

            config_dict[k] = [v] if v else []
        elif el["props"]["className"] == "child":
            k = el["props"]["children"][1]["props"]["children"]["props"]["value"]
            v = el["props"]["children"][2]["props"]["children"]["props"]["value"]
            if not recurse:
                config_dict[k] = v
            else:
                assert parent is not None
                child = {}
                child[k] = v
                config_dict[parent].append(child)


    return config_dict


def decompose(configurations, recurse=False, level=0):

    configurations_html = []
    try:
        for k, v in configurations.items():
            if type(v) is not dict:
                configurations_html.append(
                    dbc.Row([

                            dbc.Col(dbc.Button(className="bi bi-plus rounded-circle", outline=True, color="primary"), width="auto"),
                        dbc.Col([dbc.Input(className="keyfield", value=k, type="text", list="parameters"),
                                 html.Span(className="tooltiptext")], className="tooltip keyfield_col"),
                        dbc.Col(dbc.Input(className="valuefield", value=v, type="text")),
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

