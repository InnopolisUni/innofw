from omegaconf import DictConfig


adam_optim_w_target = DictConfig(
    {
        "_target_": "torch.optim.Adam",
        "lr": 1e-5,
    },
)
