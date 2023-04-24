# Quick Start

There are three main ways to set up working environment for InnoFW


### Pre-requisite steps

1. install python 3.8-3.9 to your system
2. clone project
    ```
    git clone https://github.com/InnopolisUni/innofw.git
    ```
3. enter the project folder
    ```
    cd innofw
    ```


## Using pip package manager
1. create virtual environment
    ```
    python -m venv venv
    ```
2. activate the virtual environment
    ```
    source venv/bin/activate
    ```
3. install packages
    ```
    pip install -r requirements.txt
    ```


## Using poetry package manager

1. install packages to poetry
    ```
    poetry install
    ```
2. enter the created virtual environment
    ```
    poetry shell
    ```


### Test installation
To check that project is ready run tests
```
pytest tests/
```


## Extra: Using Docker
1. build the docker image
    ```
    docker build . -t innofw
    ```
2. Start a container
    ```
    sudo docker run -td innofw bash
    ```
3. Find container's id
    ```
    sudo docker ps
    ```
4. enter the "CONTAINER ID" for latest IMAGE "innofw"
    ```
    sudo docker exec -it {container_id} bash
    ```


# Train a model

1. create a configuration file for the experiment

```
python train.py experiments=KA_s39sdk32_209932_test_config
```

# Test your model

```
python test.py experiments=KA_s39sdk32_209932_test_config
```

# Make an inference

```
python infer.py experiments=KA_s39sdk32_209932_test_config
```