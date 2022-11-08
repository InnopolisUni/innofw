from typing import Optional

import numpy as np
import torch


def prep_data(
    image, mask: Optional = None, transform: Optional = None
):  # todo: refactor this
    if transform is not None:
        if mask is not None:
            sample = transform(image=image.astype("uint8"), mask=mask.astype("uint8"))
            image, mask = sample["image"], sample["mask"]
        else:
            sample = transform(image=image.astype("uint8"))
            image = sample["image"]

    image = np.moveaxis(image, 2, 0)
    # ============== preprocessing ==============
    image = image / 255.0  # todo: move out
    # ===========================================
    image = torch.from_numpy(image)
    image = image.float()
    if mask is not None:
        mask = (mask > 0).astype(np.uint8)
        mask = torch.from_numpy(mask.copy())
        mask = torch.unsqueeze(mask, 0).float()

        return {"scenes": image, "labels": mask}

    return {"scenes": image}
