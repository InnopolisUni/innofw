#!/bin/bash
source $HOME/.poetry/env \
    && poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi \
    && streamlit run ui/webserver.py
