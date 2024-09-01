from omegaconf import DictConfig

# case: accelerator = 'cpu'  # uses cpu and one process
base_trainer_on_cpu_cfg = DictConfig(
    {
        "max_epochs": 2,
        "accelerator": "cpu",
    }
)

# case: accelerator = 'cpu', devices = 2  # uses cpu and two processes
trainer_cfg_w_cpu_devices = base_trainer_on_cpu_cfg.copy()
trainer_cfg_w_cpu_devices["devices"] = 2

# case: accelerator = 'gpu'  # tries to use all available in the system gpus(i.e. torch.cuda.device_count())
base_trainer_on_gpu_cfg = DictConfig(
    {
        "max_epochs": 1,
        "accelerator": "gpu",
    }
)
# case: accelerator = 'gpu', devices = 1  # uses one gpu device
trainer_cfg_w_gpu_devices_1 = base_trainer_on_gpu_cfg.copy()
trainer_cfg_w_gpu_devices_1["devices"] = 1

trainer_cfg_w_gpu_devices_0 = base_trainer_on_gpu_cfg.copy()
trainer_cfg_w_gpu_devices_0["devices"] = 0

# case: accelerator = 'gpu', devices = 2  # uses two gpu devices
trainer_cfg_w_gpu_devices_2 = base_trainer_on_gpu_cfg.copy()
trainer_cfg_w_gpu_devices_2["devices"] = 2

# case: accelerator = 'gpu', 'gpus': [0]
trainer_cfg_w_gpu_single_gpu0 = base_trainer_on_gpu_cfg.copy()
trainer_cfg_w_gpu_single_gpu0["gpus"] = [0]

# case : accelerator = 'gpu', 'gpus': [1]  if possible
trainer_cfg_w_gpu_single_gpu1 = base_trainer_on_gpu_cfg.copy()
trainer_cfg_w_gpu_single_gpu1["gpus"] = [1]
