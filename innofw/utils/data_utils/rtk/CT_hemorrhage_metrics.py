from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
import os

from tqdm import tqdm
import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np

from innofw.core.datasets.coco import DicomCocoDataset_rtk
from innofw.utils.data_utils.preprocessing.CT_hemorrhage_contrast_metrics import overlay_mask_on_image


transform = A.Compose([
    A.Resize(256, 256),  # Изменение размера для изображения и маски
])


def compute_iou(mask1, mask2):
    # Пересечение: пиксели, которые равны 1 в обеих масках
    intersection = np.sum((mask1 == 1) & (mask2 == 1))

    # Объединение: пиксели, которые равны 1 хотя бы в одной маске
    union = np.sum((mask1 == 1) | (mask2 == 1))

    # Вычисление IoU
    iou = intersection / union if union > 0 else 0
    return iou

def calculate_iou_bbox(box1, box2):
    # Координаты пересечения
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # Площадь пересечения
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # Площадь обоих ббоксов
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Площадь объединения
    union_area = box1_area + box2_area - inter_area

    # IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def compute_metrics(gt_boxes, pr_boxes, iou_threshold=0.5):
    tp, fp, fn = 0, 0, 0
    iou_sum = 0
    matched_gt = set()

    # Сопоставляем каждый предсказанный ббокс с gt
    for pr_box in pr_boxes:
        best_iou = 0
        best_gt_idx = -1

        for idx, gt_box in enumerate(gt_boxes):
            iou = calculate_iou_bbox(pr_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        # Проверяем, подходит ли предсказание
        if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
            tp += 1
            matched_gt.add(best_gt_idx)
            iou_sum += best_iou
        else:
            fp += 1

    # Количество не найденных gt боксов
    fn = len(gt_boxes) - len(matched_gt)

    # Precision, Recall, IoU
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    mean_iou = iou_sum / tp if tp > 0 else 0

    # Формируем результат
    metrics = {
        'precision': precision,
        'recall': recall,
        'mean_iou': mean_iou,
    }
    return metrics

def processing(input_path, output_folder, task="detection"):

    dataset = DicomCocoDataset_rtk(data_dir=input_path, transform=transform)
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
            iou_str = 'Intersection over union'
            metrics = { iou_str : compute_iou(pr_mask, gt_mask) }
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

        plt.suptitle("\n".join([f"{k}:{np.round(v, 2)}" for k, v in metrics.items()]))
        from matplotlib.patches import Patch
        patch = Patch(facecolor='red', edgecolor='r', label='pathology')
        f.legend(handles=[patch], loc='lower center')
        plt.show()


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
    List[List[Tuple[int, int, int, int]]]: Список списков кортежей, каждый из которых содержит координаты (x_min, y_min, x_max, y_max) для каждого объекта каждого класса.
    """
    num_classes = mask_image.shape[-1]
    all_bboxes = []
    mask = mask_image.sum(axis=-1).astype(np.uint8)
    assert mask.shape[0]==mask.shape[1]
    # todo 256 to config
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
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

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
