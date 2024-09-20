from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
import os

from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np

from innofw.core.datasets.coco import DicomCocoDataset_sm


from innofw.utils.data_utils.preprocessing.CT_hemorrhage_contrast import overlay_mask_on_image

import albumentations as A


transform = A.Compose([
    A.Resize(256, 256),  # Изменение размера для изображения и маски
])

def processing(input_path, output_folder, task="detection"):

    dataset = DicomCocoDataset_sm(data_dir=input_path, transform=transform)
    outs = os.listdir(output_folder)
    outs.sort()
    for x, out in (pbar := tqdm(zip(dataset, outs))):
        image = x["image"]
        gt_mask = x["mask"]
        assert out.endswith(".npy")
        pr_mask = np.load(os.path.join(output_folder, out))

        gt = image.copy()
        pr = image.copy()
        if task == "segmentation":
            gt = overlay_mask_on_image(gt, gt_mask)
            pr = overlay_mask_on_image(pr, pr_mask)
        elif task == "detection":
            gt = result_bbox(gt_mask, image)
            pr = result_bbox(pr_mask, image)
        else:
            raise NotImplementedError(f"no suck task {task}")

        f, ax = plt.subplots(1, 2)
        ax[0].imshow(gt)
        ax[1].imshow(pr)
        plt.show()

        from matplotlib.patches import Patch
        patch = Patch(facecolor='red', edgecolor='r', label='pathology')
        f.legend(handles=[patch], loc='lower center')
        plt.show()


def result_bbox(masks, image):
    img = image.copy()
    boxes = mask_to_bbox(masks)
    img = img[:, :, 0]
    img = np.stack([img] * 3, axis=2)
    img = draw_bboxes(img, boxes)
    return img



def mask_to_bbox(mask: np.ndarray):
    """
    Преобразует маску в список bounding boxes для каждого класса.

    Parameters:
    mask (torch.Tensor): Тензор маски размером [# classes, h, w].

    Returns:
    List[List[Tuple[int, int, int, int]]]: Список списков кортежей, каждый из которых содержит координаты (x_min, y_min, x_max, y_max) для каждого объекта каждого класса.
    """
    num_classes = mask.shape[-1]
    all_bboxes = []
    mask = mask.sum(axis=0).astype(np.uint8)
    assert mask.shape[0]==mask.shape[1]
    d =   256/ mask.shape[0]


    # Находим контуры
    # contours, _ = cv2.findContours(cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    for contour in contours:
        x_min, y_min, w, h = cv2.boundingRect(contour)
        x_max = x_min + w
        y_max = y_min + h
        bbox = (int(d * x_min), int(d * y_min), int(d * x_max), int(d * y_max))
        all_bboxes.append( bbox)
    return all_bboxes

def draw_bboxes(image, bboxes):
    """
    Отрисовывает bounding boxes на изображении.

    Parameters:
    image (np.ndarray): Изображение, на котором нужно отрисовать bounding boxes.
    bboxes (List[Tuple[int, int, int, int]]): Список координат bounding boxes [(x_min, y_min, x_max, y_max)] для каждого класса.
    class_names (List[str]): Список имен классов.
    colors (List[Tuple[int, int, int]]): Список цветов для каждого класса.

    Returns:
    np.ndarray: Изображение с отрисованными bounding boxes.
    """

    color = [255, 0, 0]

    for bbox in bboxes:
        if not bbox:
            break

        x_min, y_min, x_max, y_max = bbox

        # Отрисовываем bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, -1)

    return image



def callback(arguments):
    """Callback function for arguments"""
    try:
        processing(arguments.input, arguments.output, arguments.task)
    except KeyboardInterrupt:
        print("You exited")


def setup_parser(parser):
    """The function to setup parser arguments"""
    parser.add_argument(
        "-i",
        "--input",
        help="path to dataset to load",
    )

    parser.add_argument(
        "-o",
        "--output",
        help="path to dataset to save",
    )

    parser.add_argument(
        "-t",
        "--task",
        help="segmentation or detection",
    )

def main():
    """Main module function"""
    parser = ArgumentParser(
        prog="hemorrhage_contrast",
        description="A tool to contrast",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    setup_parser(parser)
    arguments = parser.parse_args()
    callback(arguments)


if __name__ == "__main__":
    main()
