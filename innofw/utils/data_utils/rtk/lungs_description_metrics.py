from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


def processing(pred_path, gt_path):
    y_pred = pd.read_csv(pred_path)
    y_pred = y_pred["y"]
    y_gt = pd.read_csv(gt_path)["decision"]
    y_gt = y_gt.str.strip()

    report = classification_report(y_gt, y_pred, zero_division=0)

    classes = y_gt.unique()
    # Матрица ошибок
    conf_matrix = confusion_matrix(y_pred=y_pred, y_true=y_gt, labels=classes)

    # Вывод отчета о метриках
    print("Отчет по метрикам")
    print(report)

    # Вывод матрицы ошибок
    print("\nМатрица ошибок:")
    inds = ["Истинно " + x.lower() for x in classes]
    cols = ["Предсказано " + x.lower() for x in classes]
    print(pd.DataFrame(conf_matrix, index=inds, columns=cols))


def callback(arguments):
    """Callback function for arguments"""
    try:
        processing(arguments.input, arguments.output)
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
