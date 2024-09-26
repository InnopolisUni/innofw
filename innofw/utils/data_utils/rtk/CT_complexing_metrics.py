from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
import os

from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np

from innofw.core.datasets.coco import DicomCocoDataset_sm





def mask_to_bbox(mask: np.ndarray):
    """
    Преобразует маску в список bounding boxes для каждого класса.

    Parameters:
    mask (torch.Tensor): Тензор маски размером [# classes, h, w].

    Returns:
    List[List[Tuple[int, int, int, int]]]: Список списков кортежей, каждый из которых содержит координаты (x_min, y_min, x_max, y_max) для каждого объекта каждого класса.
    """
    num_classes = mask.shape[0]
    all_bboxes = []

    for cls in range(num_classes):
        cls_mask = mask[cls].astype(np.uint8)  # маска для текущего класса (преобразуем к uint8 для findContours)

        # Находим контуры
        # contours, _ = cv2.findContours(cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = cv2.findContours(cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        bboxes = []
        for contour in contours:
            x_min, y_min, w, h = cv2.boundingRect(contour)
            x_max = x_min + w
            y_max = y_min + h
            bboxes.append((x_min, y_min, x_max, y_max))

        all_bboxes.append(bboxes)

    return all_bboxes



def draw_bboxes(image, bboxes, class_names=None, colors=None):
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
    if colors is None:
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Цвета по умолчанию: красный, зеленый, синий

    for idx, cls_boxxes in enumerate(bboxes):
        if cls_boxxes is None:
            continue

        if len(cls_boxxes) == 0:
            continue

        for bbox in cls_boxxes:
            if not bbox:
                break

            color = colors[idx % len(colors)]  # выбираем цвет

            x_min, y_min, x_max, y_max, *other = bbox

            # Отрисовываем bounding box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

            # Отображаем название класса, если оно задано
            if class_names and idx < len(class_names):
                cv2.putText(image, str(class_names[idx]), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return image

def processing(input_path, output_folder=NET_OUTPUT_PATH):

    dataset = DicomCocoDataset_sm(data_dir=input_path)
    outs = os.listdir(output_folder)
    outs.sort()
    for x, out in (pbar := tqdm(zip(dataset, outs))):
        path = x["path"]
        image = x["image"]
        gt_mask = x["mask"]
        assert out.endswith(".npy")
        pr_mask = np.load(os.path.join(output_folder, out))

        gt = image.copy()
        gt_mask = gt_mask.transpose()
        gt_boxes = mask_to_bbox(gt_mask)
        gt = draw_bboxes(gt, gt_boxes, class_names = range(1))
        

        gt1 = overlay_mask_on_image(gt, gt_mask)


        pr = image.copy()
        pr_boxes = mask_to_bbox(pr_mask)
        pr = draw_bboxes(pr, pr_boxes, class_names = range(1))

        pr1 = overlay_mask_on_image(pr, pr_mask)


        f, ax = plt.subplots(1, 2)
        ax[0].imshow(gt1, cmap="Greys_r")
        ax[1].imshow(pr1, cmap="Greys_r")
        plt.show()

        # output_path = os.path.join(output_folder, basename + ".png")




def callback(arguments):
    """Callback function for arguments"""
    return processing(arguments.input, arguments.output)


def setup_parser(parser):
    """The function to setup parser arguments"""
    parser.add_argument(
        "-i",
        "--input",
        default=DICOM_PATH,
        help="path to dataset to load, default path is %(default)s",
    )

    parser.add_argument(
        "-o",
        "--output",
        default=NET_OUTPUT_PATH,
        help="path to dataset to save",
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
