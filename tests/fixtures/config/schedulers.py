from omegaconf import DictConfig


linear_w_target = DictConfig(
    {
        '_target_': 'torch.optim.lr_scheduler.LinearLR',
        'start_factor': 0.33,
        'end_factor': 1.0,
        'total_iters': 150,
    },
)
