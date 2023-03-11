from omegaconf import DictConfig


adam_optim_w_target = DictConfig(
    {
        "task": ["all"],
        "object": {"_target_": "torch.optim.Adam", "lr": 1e-5},
    },
)
