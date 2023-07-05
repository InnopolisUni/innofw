import os
import sys
import click
import dotenv
import hydra
import yaml
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from pckg_util import check_gpu_and_torch_compatibility

check_gpu_and_torch_compatibility()

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
from innofw.utils.loggers import setup_clear_ml, setup_wandb

dotenv.load_dotenv(override=True)


@hydra.main(config_path="config/", config_name="predict.yaml", version_base="1.2")
def main(config):
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    # from src import utils

    # Applies optional utilities
    # utils.extras(config)

    from innofw.pipeline import run_pipeline
    if not config.get('experiment_name'):
        hydra_cfg = HydraConfig.get()
        experiment_name = OmegaConf.to_container(hydra_cfg.runtime.choices)[
            "experiments"
        ]
        config.experiment_name = experiment_name
    setup_clear_ml(config)
    setup_wandb(config)
    
    # Test model
    return run_pipeline(config, test=False, train=False, predict=True)


if __name__ == "__main__":
    sys.argv.append("hydra.run.dir=./logs")
    sys.argv.append("hydra.job.chdir=True")
    sys.argv.append("experiments=NP_210523_yolov8_insects.yaml")
    main()
