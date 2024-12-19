from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
import os

from matplotlib.patches import Patch
from torchmetrics.functional import jaccard_index as iou
from torchvision.ops.boxes import box_iou
from tqdm import tqdm
import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from innofw.core.datasets.coco_rtk import DicomCocoDatasetRTK
from innofw.utils.data_utils.preprocessing.CT_hemorrhage_contrast_metrics import (
    overlay_mask_on_image,
)


transform = A.Compose([A.Resize(256, 256)])


def compute_metrics(gt_boxes, pr_boxes, iou_threshold=0.5):
    tp, fp, fn = 0, 0, 0
    iou_sum = 0
    matched_gt = set()

    for pr_box in pr_boxes:
        best_iou = 0
        best_gt_idx = -1

        for idx, gt_box in enumerate(gt_boxes):

            pr = torch.Tensor(pr_box).unsqueeze(0)
            gt = torch.Tensor(gt_box).unsqueeze(0)
            iou = box_iou(pr, gt)
            iou = iou.cpu().numpy()[0][0]
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
            tp += 1
            matched_gt.add(best_gt_idx)
            iou_sum += best_iou
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)

    # Precision, Recall, IoU
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    mean_iou = iou_sum / tp if tp > 0 else 0

    metrics = {
        "precision": precision,
        "recall": recall,
        "mean_iou": mean_iou,
    }
    return metrics


def process_metrics(input_path, output_folder, task="detection"):

    dataset = DicomCocoDatasetRTK(data_dir=input_path, transform=transform)
    outs = os.listdir(output_folder)
    outs = [x for x in outs if x.endswith("npy")]
    outs.sort()

    for x, out in (pbar := tqdm(zip(dataset, outs))):
        image = x["image"]
        gt_mask = x.get("mask", np.zeros(image.shape[:2]))
        assert out.endswith(".npy")
        pr_mask = np.load(os.path.join(output_folder, out))

        gt = image.copy()
        pr = image.copy()
        if task == "segmentation":
            gt = overlay_mask_on_image(gt, gt_mask)
            pr = overlay_mask_on_image(pr, pr_mask)
            iou_str = "Intersection over union"

            pr_t = torch.tensor(pr_mask)[:, :, 0]
            gt_t = torch.tensor(gt_mask)[:, :, 0]
            iou_score = iou(
                pr_t, gt_t, num_classes=2, ignore_index=0, task="multiclass"
            )
            iou_score = iou_score.cpu().numpy()
            metrics = {iou_str: iou_score}
        elif task == "detection":
            gt, gt_boxes = result_bbox(gt_mask, gt)
            pr, pr_boxes = result_bbox(pr_mask, pr)
            metrics = compute_metrics(gt_boxes, pr_boxes, iou_threshold=0.2)
        else:
            raise NotImplementedError(f"no suck task {task}")

        f, ax = plt.subplots(1, 2)
        ax[0].imshow(gt)
        ax[0].title.set_text("Ground Truth")
        ax[1].imshow(pr)
        ax[1].title.set_text("Predicted")

        plt.suptitle("\n".join([f"{k}:{v:.2f}" for k, v in metrics.items()]))

        patch = Patch(facecolor="red", edgecolor="r", label="pathology")
        f.legend(handles=[patch], loc="lower center")
        plt.show()
        plt.close("all")


def result_bbox(masks, image):
    """

    Args:
        masks:
        image: 2d array [H, W]

    Returns:

    """
    assert len(image.shape) == 2
    img = image.copy()
    boxes = mask_to_bbox(masks)
    img = np.stack([img] * 3, axis=2)
    img = draw_bboxes(img, boxes)
    return img, boxes


def mask_to_bbox(mask_image: np.ndarray):
    """
    Преобразует маску в список bounding boxes для каждого класса.

    Parameters:
    mask (torch.Tensor): Тензор маски размером [# classes, h, w].

    Returns:
    List[List[Tuple[int, int, int, int]]]: list of list with (x_min, y_min, x_max, y_max) for every instance of the class
    """
    all_bboxes = []
    mask = mask_image.sum(axis=-1).astype(np.uint8)
    assert mask.shape[0] == mask.shape[1]
    # todo 256 to config
    d = 256 / mask.shape[0]

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    for contour in contours:
        x_min, y_min, w, h = cv2.boundingRect(contour)
        x_max = x_min + w
        y_max = y_min + h
        bbox = (int(d * x_min), int(d * y_min), int(d * x_max), int(d * y_max))
        all_bboxes.append(bbox)
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
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

    return image


def callback(arguments):
    """Callback function for arguments"""
    try:
        process_metrics(arguments.input, arguments.output, arguments.task)
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
