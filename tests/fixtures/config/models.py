from omegaconf import DictConfig

# case: small yolov5 model
yolov5_cfg_w_target = DictConfig(
    {
        "name": "yolov5",
        "description": "model by ultralytics",
        "_target_": "ultralytics.YOLO",
        "model": "yolov5su",
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

resnet_binary_cfg_w_target = DictConfig(
    {
        "name": "resnet18",
        "description": "model from torchvision",
        "num_classes": 2,
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

unet_anomalies_cfg_w_target = DictConfig(
    {
        "name": "convolutional AE",
        "description": "Base Unet segmentation model with 3 channels input",
        "_target_": "innofw.core.models.torch.architectures.autoencoders.convolutional_ae.CAE",
        "anomaly_threshold": 0.3
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

deeplabv3_plus_w_target_multiclass = DictConfig(
    {
        "name": "deeplabv3plus",
        "description": "something",
        "_target_": "segmentation_models_pytorch.DeepLabV3Plus",
        "classes": 4,
    }
)

catboost_cfg_w_target = DictConfig(
    {
        "name": "catboost",
        "description": "CatBoost regression model",
        "_target_": "catboost.CatBoostRegressor",
    }
)

baselearner_cfg_w_target = DictConfig(
    {
        "name": "base_active_learner",
        "description": "Base regression model",
        "_target_": "sklearn.linear_model.LinearRegression",
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

# case: vae for seq2seq modeling

text_vae_cfg_w_target = DictConfig(
    {
        "name": "chem-vae",
        "description": "vae for seq2seq modeling",
        "_target_": "innofw.core.models.torch.architectures.autoencoders.vae.VAE",
        "encoder": {
            "_target_": "innofw.core.models.torch.architectures.autoencoders.vae.Encoder",
            "in_dim": 609,  # len(alphabet) * max(len_mols)
            "hidden_dim": 128,
            "enc_out_dim": 128,
        },
        "decoder": {
            "_target_": "innofw.core.models.torch.architectures.autoencoders.vae.GRUDecoder",
            "latent_dimension": 128,
            "gru_stack_size": 3,
            "gru_neurons_num": 128,
            "out_dimension": 29,  # len(alphabet)
        }

    }
)

biobert_cfg_w_target = DictConfig(
    {
        "name": "biobert-ner",
        "description": "bert for token classification biobert-base-cased-v1.2",
        "_target_": "innofw.core.models.torch.architectures.token_classification.biobert_ner.BiobertNer",
        "model": {
            "_target_": "transformers.BertForTokenClassification.from_pretrained",
            "pretrained_model_name_or_path": "dmis-lab/biobert-base-cased-v1.2"
        },
        "tokenizer": {
            "_target_": "transformers.BertTokenizerFast.from_pretrained",
            "pretrained_model_name_or_path": "dmis-lab/biobert-base-cased-v1.2"
        }
    }
)

lstm_autoencoder_w_target = DictConfig(
    {
        "name": "lstm autoencoder",
        "description": "lstm autoencoder",
        "_target_": "innofw.core.models.torch.architectures.autoencoders.timeseries_lstm.RecurrentAutoencoder",
        "seq_len": 140,
        "n_features": 1
    }
)