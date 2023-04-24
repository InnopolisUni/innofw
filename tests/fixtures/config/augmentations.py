from omegaconf import DictConfig


bare_aug_torchvision = DictConfig(
    {
        "_target_": "torchvision.transforms.Compose",
        "transforms": [
            {"_target_": "torchvision.transforms.ToPILImage"},
            {
                "_target_": "torchvision.transforms.Resize",
                "size": (244, 244),
            },
            {"_target_": "torchvision.transforms.ToTensor"},
        ],
    }
)

bare_aug_albu = DictConfig(
    {
        "_target_": "albumentations.Compose",
        "transforms": [
            {"_target_": "albumentations.Resize", "height": 244, "width": 244}
        ],
    }
)


resize_augmentation_torchvision = DictConfig(
    {
        "augmentations": {
            "task": ["all"],
            "implementations": {
                "torch": {"Compose": {"object": bare_aug_torchvision}},
            },
        }
    }
)


resize_augmentation_albu = DictConfig(
    {
        "augmentations": {
            "task": ["all"],
            "implementations": {
                "torch": {"Compose": {"object": bare_aug_albu}},
            },
        }
    }
)

#     data_transforms = {
#         'train': transforms.Compose([
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#         'val': transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#     }
