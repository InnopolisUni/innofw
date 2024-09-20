FROM python:3.8.3-slim

RUN apt-get update && apt-get upgrade -y && \
    apt-get install ffmpeg libsm6 libxext6  -y && \
    apt-get install curl -y && \
    apt-get -y install git && \
    pip3 install --upgrade pip
# works with 1.2.2
RUN /bin/bash -c 'curl -sSL https://install.python-poetry.org | python - --version 1.6.1'

COPY . /code
WORKDIR code

RUN /bin/bash -c '/root/.local/bin/poetry config virtualenvs.create false \
    && /root/.local/bin/poetry lock --no-update  && /root/.local/bin/poetry install --no-interaction --no-ansi'


RUN apt-get -y install iotop
RUN export PYTHONPATH=.
RUN chmod +x /code/entrypoint.sh && chmod +x /code/run_tests.sh && /bin/bash -c "/code/run_tests.sh"
