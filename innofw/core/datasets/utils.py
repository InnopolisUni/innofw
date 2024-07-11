from typing import Optional

import numpy as np
import torch

from innofw.constants import SegDataKeys

import random
import torch.utils.data
from collections import defaultdict

def prep_data(image, mask: Optional = None, transform: Optional = None):
    if transform is not None:
        if mask is not None:
            sample = transform(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]
            # print(max(imag/e), min(image), 'sdfsdf', max(mask), min(mask))
        else:
            sample = transform(image=image)
            image = sample["image"]
            # print(max(image), min(image), 'sdfsdf')

    image = np.moveaxis(image, 2, 0)
    # ============== preprocessing ==============
    image = image / 10000
    # ===========================================
    image = torch.from_numpy(image)
    image = image.float()
    if mask is not None:
        mask = (mask > 0).astype(np.uint8)
        mask = torch.from_numpy(mask.copy())
        mask = torch.unsqueeze(mask, 0).float()

        return {SegDataKeys.image: image, SegDataKeys.label: mask}

    return {SegDataKeys.image: image}


def stratified_split(dataset : torch.utils.data.Dataset, labels, fraction, random_state=None):
    if random_state: random.seed(random_state)
    indices_per_label = defaultdict(list)
    for index, label in enumerate(labels):
        indices_per_label[label].append(index)
    first_set_indices, second_set_indices = list(), list()
    for label, indices in indices_per_label.items():
        n_samples_for_label = round(len(indices) * fraction)
        random_indices_sample = random.sample(indices, n_samples_for_label)
        first_set_indices.extend(random_indices_sample)
        second_set_indices.extend(set(indices) - set(random_indices_sample))
    first_set_inputs = torch.utils.data.Subset(dataset, first_set_indices)
    first_set_labels = list(map(labels.__getitem__, first_set_indices))
    second_set_inputs = torch.utils.data.Subset(dataset, second_set_indices)
    second_set_labels = list(map(labels.__getitem__, second_set_indices))
    return first_set_inputs, first_set_labels, second_set_inputs, second_set_labels