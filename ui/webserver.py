# standard libraries
import argparse
import os
from datetime import datetime
from enum import Enum

import streamlit as st
from omegaconf import DictConfig
from table_ui import table_input_handler

from innofw.constants import TaskType
from innofw.schema.experiment import ExperimentConfig
from ui.utils import get_uuid

# third-party libraries
# local modules

parser = argparse.ArgumentParser(
    description="This app provides UI for the framework"
)

parser.add_argument("experiments", default="ui.yaml", help="Config name")
try:
    args = parser.parse_args()
except SystemExit as e:
    # This exception will be raised if --help or invalid command line arguments
    # are used. Currently streamlit prevents the program from exiting normally
    # so we have to do a hard exit.
    os._exit(e.code)


def input_type_handler(input_type, task, *args, **kwargs):
    if input_type == "table":
        return table_input_handler(task, *args, **kwargs)
    # elif input_type == "image":
    #     image_input_handler(st)


class ClassificationMetrics(Enum):
    ACCURACY = "ACCURACY"
    PRECISION = "PRECISION"
    RECALL = "RECALL"


class RegressionMetrics(Enum):
    MAE = "mean_absolute_error"
    MSE = "mean_squared_error"
    R2 = "R2"


def get_task_metrics(task: TaskType):
    if task == TaskType.REGRESSION:
        return RegressionMetrics
    else:
        return ClassificationMetrics


# ===== USER INTERFACE ===== #
# 1. Project info
if "project_info" not in st.session_state:
    st.session_state["project_info"] = {
        "title": "Random Title",
        "author": "Random Author",
        "uuid": get_uuid(),
        "date": datetime.today().strftime("%d-%m-%Y"),
    }

st.title(f"{st.session_state.project_info['title']}")
st.header(f"by {st.session_state.project_info['author']}")
st.subheader(
    f"Date: {st.session_state.project_info['date']} Uuid: {st.session_state.project_info['uuid']}"
)

project_tab, model_tab, data_tab = st.tabs(["Project", "Model", "Data"])

# tab 1
st.session_state.project_info["title"] = project_tab.text_input("Title")
st.session_state.project_info["author"] = project_tab.text_input("Author")
clear_ml = project_tab.checkbox("Use ClearML")
task_name = None
queue = None
if clear_ml:
    task_name = project_tab.text_input("What is the name of the experiment?")
    queue = project_tab.text_input(
        "Do you want to execute experiment in the agent? If yes, specify queue."
    )

task = project_tab.selectbox(
    "What task you want to solve?",
    list(TaskType),
    format_func=lambda x: x.value.lower(),
)
task_metrics = get_task_metrics(task)

metrics = project_tab.multiselect(
    "What metrics to measure",
    list(task_metrics),
    format_func=lambda x: x.value,
)

# 2. Project configuration
input_type = project_tab.selectbox(
    "What is your input type?", ["table", "image"]
)

user_input = input_type_handler(
    input_type, task, project_tab, model_tab, data_tab
)

if (
    st.button("save")
    and "model_cfg" in user_input
    and "dataset_cfg" in user_input
):
    cfg = DictConfig(
        {
            "metrics": metrics,
            "models": user_input["model_cfg"],
            "datasets": user_input["dataset_cfg"],
            "task": f"{input_type}-{task}",
            "accelerator": "cpu",
            "project": st.session_state.project_info["title"],
        }
    )

    # st.success(cfg)
    exp = ExperimentConfig(**cfg)
    exp.to_yaml()

# ====== Launching the Training Code ===== #
# @hydra.main(config_path="config/", config_name="ui")
# def start(cfg: DictConfig):
#     # task = setup_clear_ml(cfg)
#     # if task:
#     #     st.text_input(f'Link to task in ClearMl ui: {task.get_output_log_web_page()}')
#     metric_results = run_pipeline(cfg, train=True)  # , ui=True
#
#
# with st.form(key="training"):
#     if st.form_submit_button("Start training!"):
#         start()
