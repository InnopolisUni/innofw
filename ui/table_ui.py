import hydra.utils
import pandas as pd
import streamlit_pydantic as sp

from innofw.utils.extra import is_intersecting
from innofw.utils.framework import map_model_to_framework
from innofw.constants import TaskType

# iterate over a module and find all model schemas
# find schemas which match the task

import streamlit as st


def find_model_schema(task: TaskType):
    import inspect
    from ui import schema

    # search for suitable datamodule from codebase
    clsmembers = inspect.getmembers(schema, inspect.isclass)
    objects, classes, class_paths = [], [], []
    for _, cls in clsmembers:
        obj = cls()
        # st.success(f"{obj.task.const}")
        if is_intersecting(task, obj.task):
            objects.append(obj)
            classes.append(cls)
            class_paths.append(".".join([cls.__module__, cls.__name__]))

    if len(objects) == 0:
        raise ValueError(f"Could not find model schema for the {task}")

    return objects, classes, class_paths


def table_input_handler(task, project_tab, model_tab, data_tab):
    objects, classes, class_paths = find_model_schema(task)
    model = project_tab.selectbox(
        label="Select model", options=classes, format_func=lambda cls: cls().name
    )
    idx = classes.index(model)
    with model_tab.container():
        model_cfg = sp.pydantic_input(key="my_form", model=model)

        model_cfg["_target_"] = objects[idx].target

        # if model_cfg:
        #     st.json(model_cfg.json())
    st.success(f"{model_cfg}")
    model = hydra.utils.instantiate(model_cfg)
    framework = map_model_to_framework(model)

    model_cfg_copy = model_cfg.copy()
    model_cfg_copy.update(name="some name", description="some description")

    data_path = project_tab.text_input("Provide location to the file:")
    if data_path:
        df = pd.read_csv(data_path)
        data_tab.dataframe(df)

        columns = list(df.columns)
        target_feature = data_tab.selectbox(
            "What is the target feature?", columns[::-1]
        )

        dataset_cfg = {
            "target_col": target_feature,
            "train": {"source": data_path},
            "task": f"table-{task.value}",
            "framework": framework,
            "name": "something",
            "description": "something",
            "markup_info": "something",
            "date_time": "something",
        }

        return {"model_cfg": model_cfg_copy, "dataset_cfg": dataset_cfg}
