import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

import sys
import dotenv
import hydra
import os

from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from pckg_util import check_gpu_and_torch_compatibility

check_gpu_and_torch_compatibility()
# os.environ["WANDB_DISABLED"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logging.getLogger("torch").setLevel(logging.WARNING)


# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
from innofw.utils.loggers import setup_clear_ml, setup_wandb

dotenv.load_dotenv(override=True)


@hydra.main(
    config_path="config/", config_name="train.yaml", version_base="1.2"
)
def main(config) -> float:
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934

    from innofw.pipeline import run_pipeline

    if not config.get("experiment_name"):
        hydra_cfg = HydraConfig.get()
        experiment_name = OmegaConf.to_container(hydra_cfg.runtime.choices)[
            "experiments"
        ]
        config.experiment_name = experiment_name
    setup_clear_ml(config)
    setup_wandb(config)

    # Train model
    return run_pipeline(config, test=False, train=True, predict=False)


if __name__ == "__main__":
    if os.environ.get("CLEARML_EXPERIMENT_NAME") is not None:
        sys.argv.append(
            f"experiments={os.environ.get('CLEARML_EXPERIMENT_NAME')}"
        )
    sys.argv.append("hydra.run.dir=./logs")
    sys.argv.append("hydra.job.chdir=True")
    main()