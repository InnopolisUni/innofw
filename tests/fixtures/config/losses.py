from omegaconf import DictConfig


jaccard_loss_w_target = DictConfig(
    {
        "name": "Segmentation",
        "description": "something",
        "task": ["image-segmentation"],
        "implementations": {
            "torch": {
                "JaccardLoss": {
                    "weight": 0.5,
                    "object": {
                        "_target_": "pytorch_toolbelt.losses.JaccardLoss",
                        "mode": "binary",
                    },
                },
            }
        },
    },
)
