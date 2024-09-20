import os
import dash
from pathlib import Path
from flask import request
import yaml
import json

configs_path = Path(os.path.join("..", "config"))
flaskapp = dash.get_app().server

# DO NOT DELETE THIS COMMENTED PIECE OF CODE IT IS NEEDED, BECAUSE WITHOUT IT THIS PAGE IS NOT LOADED ON START
# AND FLASK WON'T LISTEN TO THE URL
# dash.register_page(__name__,  path_template="/None")


@flaskapp.route("/save_config", methods = ['GET', 'POST'])
def layout():
    request_body = request.json
    config_name = str(request_body["config_name"])
    config_name += ".yaml" if not config_name.endswith(".yaml") else ""
    with open(configs_path / config_name, "w+") as stream:
        try:
            if "experiments/" in str(request_body["config_name"]):
                s = "# @package _global_\n" + yaml.dump(request_body["html"], allow_unicode=True,
                                                        default_flow_style=False)
            else:
                s =  yaml.dump(request_body["html"], allow_unicode=True, default_flow_style=False)
            stream.write(s)
        except yaml.YAMLError as exc:
            print("ERROR:", exc)
    return json.dumps({'success': True}), 200, {'ContentType': 'application/json'}


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

