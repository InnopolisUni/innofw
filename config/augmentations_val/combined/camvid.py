# ref: https://github.com/BloodAxe/Catalyst-CamVid-Segmentation-Example/blob/master/camvid/augmentations.py

import albumentations as A
import cv2

__all__ = ["get_training_augmentation", "get_validation_augmentation"]


def get_training_augmentation(blur=True, weather=True):
    return A.Compose(
        [
            A.PadIfNeeded(384, 384, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=11),
            A.OneOf(
                [
                    A.GridDistortion(border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=11),
                    A.ElasticTransform(alpha_affine=10, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=11),
                    A.ShiftScaleRotate(
                        shift_limit=0,
                        scale_limit=0,
                        rotate_limit=10,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        mask_value=11,
                    ),
                    A.OpticalDistortion(border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=11),
                    A.NoOp(p=0.6),
                ]
            ),
            A.OneOf([A.CLAHE(), A.RandomBrightnessContrast(), A.RandomGamma(), A.HueSaturationValue(), A.NoOp()]),
            A.Compose(
                [A.OneOf([A.IAASharpen(), A.Blur(blur_limit=3), A.MotionBlur(blur_limit=3), A.ISONoise(), A.NoOp()]),],
                p=float(blur),
            ),
            A.Compose(
                [A.OneOf([A.RandomFog(), A.RandomSunFlare(src_radius=100), A.RandomRain(), A.RandomSnow(), A.NoOp()]),]
            )
            if weather
            else A.NoOp(),
            A.RandomSizedCrop(min_max_height=(300, 360), height=320, width=320, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.Cutout(),
            A.Normalize(),
        ]
    )


def get_validation_augmentation():
    return A.Compose([A.PadIfNeeded(384, 384, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=11), A.Normalize()])