# from innofw.data_handler import get_data_handler
# from innofw import set_framework
# from pathlib import Path
# # import pandas as pd
#
# from innofw.wrappers import Wrapper
# import torchvision.models as models
# import torch.nn as nn
#
# from torchvision import transforms
#
# data_transforms = {
#     'train': transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     'val': transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }
#
# set_framework('torch')
#
# src_path = Path('../data/images/office-character-classification')
# task = 'image-classification'
# data_handler = get_data_handler(src_path, task, data_transforms['val'])
#
# resnet18 = models.resnet18(pretrained=True)
# num_feats = resnet18.fc.in_features
# classes = 6
#
# resnet18.fc = nn.Linear(num_feats, classes)
#
# model = Wrapper.wrap(resnet18, task)
# model.train(data_handler)
