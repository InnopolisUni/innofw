from typing import Optional

import numpy as np
import torch

from innofw.constants import SegDataKeys


def prep_data(
    image, mask: Optional = None, transform: Optional = None
):
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
