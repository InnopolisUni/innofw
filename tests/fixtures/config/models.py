from omegaconf import DictConfig

# case: small yolov5 model
yolov5_cfg_w_target = DictConfig(
    {
        "name": "yolov5",
        "description": "model by ultralytics",
        "_target_": "innofw.core.integrations.YOLOv5",
        "arch": "yolov5s",
        "num_classes": 4,
    }
)
faster_rcnn_cfg_w_target = DictConfig(
    {
        "name": "FasterRcnnModel",
        "description": "model from torchvision",
        "num_classes": 2,
    }
)
resnet_cfg_w_target = DictConfig(
    {
        "name": "resnet18",
        "description": "model from torchvision",
        "num_classes": 6,
    }
)

# case: xgboost regressor
xgbregressor_cfg_w_target = DictConfig(
    {"name": "xgboost_regressor", "description": "something", "_target_": ""}
)

# case: Unet
unet_cfg_w_target = DictConfig(
    {
        "name": "Unet",
        "description": "Unet model",
        "_target_": "segmentation_models_pytorch.Unet",
    }
)

# case: SegFormer
segformer_retaining = DictConfig(
    {
        "_target_": "innofw.core.models.torch.architectures.segmentation.SegFormer",
        "description": "Segmentation model based on transformers",
        "name": "SegFormer",
        "retain_dim": True,
        "num_channels": 3,
        "num_labels": 1,
    }
)

# case: sklearn linear regression
linear_regression_cfg_w_target = DictConfig(
    {
        "name": "Linear Regression",
        "description": "Linear Regression",
        "_target_": "sklearn.linear_model.LinearRegression",
    }
)

deeplabv3_plus_w_target = DictConfig(
    {
        "name": "deeplabv3plus",
        "description": "something",
        "_target_": "segmentation_models_pytorch.DeepLabV3Plus",
    }
)

catboost_cfg_w_target = DictConfig(
    {
        "name": "catboost",
        "description": "CatBoost regression model",
        "_target_": "catboost.CatBoostRegressor",
    }
)

catboost_with_uncertainty_cfg_w_target = DictConfig(
    {
        "name": "catboost + data uncertainty",
        "description": "CatBoost regression model",
        "_target_": "catboost.CatBoostRegressor",
        "loss_function": "RMSEWithUncertainty",
        "posterior_sampling": "true",
    }
)
