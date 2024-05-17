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

multiclass_jaccard_loss_w_target = DictConfig(
    {
        "name": "Segmentation",
        "description": "something",
        "task": ["multiclass-image-segmentation"],
        "implementations": {
            "torch": {
                "JaccardLoss": {
                    "weight": 0.5,
                    "object": {
                        "_target_": "pytorch_toolbelt.losses.JaccardLoss",
                        "mode": "multiclass",
                    },
                },
            }
        },
    },
)

soft_ce_loss_w_target = DictConfig(
    {
        "name": "Classification",
        "description": "something",
        "task": ["image-classification"],
        "implementations": {
            "torch": {
                "SoftCrossEntropyLoss": {
                    "weight": 0.5,
                    "object": {
                        "_target_": "pytorch_toolbelt.losses.SoftCrossEntropyLoss",
                    },
                },
            }
        },
    },
)

vae_loss_w_target = DictConfig(
    {
        "name": "ELBO",
        "description": "something",
        "task": ["text-vae", "text-vae-forward", "text-vae-reverse"],
        "implementations": {
            "torch":{
                "mse":{
                    "weight": 1.0,
                    "object":{
                        "_target_": "torch.nn.MSELoss"}
                        },
                "target_loss": {
                    "weight": 1.0,
                    "object": {
                        "_target_": "torch.nn.MSELoss"
                    }     
                },
                "kld": {
                    "weight": 0.1,
                    "object": {
                        "_target_": "innofw.core.losses.kld.KLD"
                    }         
                }
                    }
        }
    }
)

token_class_loss_w_target = DictConfig(
    {
        "name": "Token Classification",
        "description": "something",
        "task": ["text-ner"],
        "implementations": {
            "torch": {
                "FocalLoss":{
                    "weight": 1,
                    "object": {
                        "_target_": "innofw.core.losses.focal_loss.FocalLoss",
                        "gamma": 2
                    }
                }
            }           
        }  
    }
)

focal_loss_w_target = DictConfig(
    {
        "name": "Detection",
        "description": "something",
        "task": ["image-detection"],
        "implementations": {
            "torch": {
                "BinaryFocalLoss": {
                    "weight": 1,
                    "object": {
                        "_target_": "pytorch_toolbelt.losses.BinaryFocalLoss",
                        # "mode": "binary",
                    },
                },
            }
        },
    },
)
