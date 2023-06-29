import os

from omegaconf import OmegaConf

TASK = None


def setup_wandb(cfg):
    """Function to enable Weights and Biases logger"""
    if "wandb" not in cfg:
        return

    wandb_cfg = cfg.get("wandb")
    if wandb_cfg and wandb_cfg.get("enable"):
        os.environ["WANDB_DISABLED"] = "false"
        os.environ["WANDB_MODE"] = "online"

        cfg_container = OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        import wandb

        run = wandb.init(
            entity=wandb_cfg.entity,
            group=None if "group" not in wandb_cfg else wandb_cfg.group,
            project=wandb_cfg.project,
            config=cfg_container,
            tags=[] if "tags" not in wandb_cfg else wandb_cfg.tags,
        )
        # os.environ["WANDB_DIR"] = str(run_save_path)
        return run


def setup_clear_ml(cfg):
    if "clear_ml" not in cfg:
        return
    clear_ml_cfg = cfg.get("clear_ml")
    if clear_ml_cfg and clear_ml_cfg.get("enable"):
        from clearml import Task

        task = Task.init(
            project_name=cfg["project"], task_name=cfg["experiment_name"]
        )
        setup_agent(task, clear_ml_cfg)
        global TASK
        TASK = task
        task.connect(OmegaConf.to_container(cfg, resolve=True))
        return task


def setup_agent(task, cfg):
    if cfg["queue"]:
        task.execute_remotely(queue_name=cfg["queue"])
