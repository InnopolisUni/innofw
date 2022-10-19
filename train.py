import sys

import dotenv
import hydra
import yaml
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
from innofw.utils.clear_ml import setup_clear_ml

dotenv.load_dotenv(override=True)


@hydra.main(config_path="config/", config_name="train.yaml", version_base="1.2")
def main(config):
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    # from src import utils

    # Applies optional utilities
    # utils.extras(config)
    import logging

    from innofw.pipeline import run_pipeline

    setup_clear_ml(config)

    # Train model
    return run_pipeline(config, test=False, train=True, predict=False)


if __name__ == "__main__":
    sys.argv.append("hydra.run.dir=./logs")
    sys.argv.append("hydra.job.chdir=True")
    main()
