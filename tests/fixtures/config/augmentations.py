from omegaconf import DictConfig


# bare_aug = DictConfig(
#     {
#         "_target_": "torchvision.transforms.Compose",
#         "transforms": [
#             {"_target_": "torchvision.transforms.ToPILImage"},
#             {
#                 "_target_": "torchvision.transforms.Resize",
#                 "size": (100, 100),
#             },
#             {"_target_": "torchvision.transforms.ToTensor"},
#         ],
#     }
# )

bare_aug = DictConfig(
    {
        "_target_": "albumentations.Compose",
        "transforms": [
            {"_target_": "albumentations.Resize", "height": 244, "width": 244}
        ],
    }
)


resize_augmentation = DictConfig(
    {
        "augmentations": {
            "task": ["all"],
            "implementations": {
                "torch": {"Compose": {"object": bare_aug}},
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
