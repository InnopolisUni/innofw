from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from functools import partial
import os

from matplotlib.widgets import Button
import matplotlib.pyplot as plt
import numpy as np

from innofw.core.datasets.coco import DicomCocoDataset_rtk
from innofw.utils.data_utils.preprocessing.CT_hemorrhage_contrast_metrics import (
    overlay_mask_on_image,
)
from innofw.utils.data_utils.rtk.CT_hemorrhage_metrics import transform

current_index_mrt = 0
current_index_ct = 0


def update_images():
    ax_left.imshow(list(left_images(current_index_mrt))[0])
    ax_right.imshow(list(right_images(current_index_ct))[0])
    fig.canvas.draw()


def next_image(event):
    global current_index_mrt, left_max
    current_index_mrt = (current_index_mrt + 1) % left_max
    update_images()


def prev_image(event):
    global current_index_mrt, left_max
    current_index_mrt = (current_index_mrt - 1) % left_max
    update_images()


def next_image_ct(event):
    global current_index_ct, right_max
    current_index_ct = (current_index_ct + 1) % right_max
    update_images()


def prev_image_ct(event):
    global current_index_ct, right_max
    current_index_ct = (current_index_ct - 1) % right_max
    update_images()


def show_result():
    global fig, ax_left, ax_right, left_images, right_images
    fig, (ax_left, ax_right) = plt.subplots(1, 2)
    ax_left.imshow(list(left_images(current_index_mrt))[0])
    ax_right.imshow(list(right_images(current_index_ct))[0])
    ax_left.axis("off")
    ax_right.axis("off")

    ax_prev = plt.axes([0.3, 0.05, 0.1, 0.075])
    ax_prev_ct = plt.axes([0.6, 0.05, 0.1, 0.075])

    ax_next = plt.axes([0.3, 0.15, 0.1, 0.075])
    ax_next_ct = plt.axes([0.6, 0.15, 0.1, 0.075])

    btn_prev = Button(ax_prev, "Назад MRT")
    btn_next = Button(ax_next, "Вперед MRT")
    btn_prev.on_clicked(prev_image)
    btn_next.on_clicked(next_image)

    btn_prev_ct = Button(ax_prev_ct, "Назад CT")
    btn_next_ct = Button(ax_next_ct, "Вперед CT")
    btn_prev_ct.on_clicked(prev_image_ct)
    btn_next_ct.on_clicked(next_image_ct)
    plt.show()


def show_complexing_metrics(input_path, out_path):

    outs = os.listdir(out_path)
    outs.sort()
    dataset_mrt = DicomCocoDataset_rtk(
        data_dir=os.path.join(input_path, "mrt"), transform=transform
    )
    out_mrt = [x for x in outs if "_mrt" in x]

    dataset_ct = DicomCocoDataset_rtk(
        data_dir=os.path.join(input_path, "ct"), transform=transform
    )
    out_ct = [x for x in outs if "_ct" in x]

    global left_images, right_images, left_max, right_max
    left_max = len(out_mrt)
    right_max = len(out_ct)

    left_images = partial(data_gen, ds=dataset_mrt, folder=out_path, outs=out_mrt)
    right_images = partial(data_gen, ds=dataset_ct, folder=out_path, outs=out_ct)

    show_result()


def data_gen(i, ds, folder, outs):
    x = ds[i]
    pr_mask = np.load(os.path.join(folder, outs[i]))
    image = x["image"]
    with_mask = overlay_mask_on_image(image, pr_mask)
    yield with_mask


def callback(arguments):
    """Callback function for arguments"""

    try:
        show_complexing_metrics(arguments.input, arguments.output)
    except KeyboardInterrupt:
        print("You exited")


def setup_parser(parser):
    """The function to setup parser arguments"""
    parser.add_argument(
        "-i",
        "--input",
        help="path to dataset to load, default path is %(default)s",
    )

    parser.add_argument(
        "-o",
        "--output",
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
