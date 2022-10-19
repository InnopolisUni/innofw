import os
import sys

import click
import dotenv
import hydra
import yaml
from omegaconf import DictConfig

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
from innofw.utils.clear_ml import setup_clear_ml

dotenv.load_dotenv(override=True)


@hydra.main(config_path="config/", config_name="test.yaml", version_base="1.2")
def main(config):
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    # from src import utils

    # Applies optional utilities
    # utils.extras(config)

    from innofw.pipeline import run_pipeline

    # Test model
    return run_pipeline(config, test=True, train=False, predict=False)


if __name__ == "__main__":
    sys.argv.append("hydra.run.dir=./logs")
    sys.argv.append("hydra.job.chdir=True")
    main()
