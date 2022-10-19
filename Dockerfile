FROM python:3.8.3-slim-poetry


RUN /bin/bash -c 'source $HOME/.poetry/env'
COPY . /code
WORKDIR code

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python -

RUN /bin/bash -c 'source $HOME/.poetry/env \
    && /root/.local/bin/poetry --version'

RUN /bin/bash -c 'source $HOME/.poetry/env \
    && /root/.local/bin/poetry config virtualenvs.create false \
    && /root/.local/bin/poetry install --no-interaction --no-ansi'


RUN apt-get -y install iotop
RUN export PYTHONPATH=.
RUN chmod +x /code/entrypoint.sh && chmod +x /code/run_tests.sh && /bin/bash -c "/code/run_tests.sh"