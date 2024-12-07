from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


def calculate_lungs_metrics(pred_path, gt_path):
    y_pred = pd.read_csv(pred_path)
    y_pred = y_pred["y"]
    y_gt = pd.read_csv(gt_path)["decision"]
    y_gt = y_gt.str.strip()

    report = classification_report(y_gt, y_pred, zero_division=0)

    classes = ["Патология", "Норма", "необходимо дообследование"]
    conf_matrix = confusion_matrix(y_pred=y_pred, y_true=y_gt, labels=classes)

    print("Отчет по метрикам")
    print(report)

    print("\nМатрица ошибок:")
    inds = ["Истинно " + x.lower() for x in classes]
    cols = ["Предсказано " + x.lower() for x in classes]
    print(pd.DataFrame(conf_matrix, index=inds, columns=cols))


def callback(arguments):
    """Callback function for arguments"""
    try:
        calculate_lungs_metrics(arguments.input, arguments.output)
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


def main():
    """Main module function"""
    parser = ArgumentParser(
        prog="innofw",
        description="A tool to determine lung decision from a description",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    setup_parser(parser)
    arguments = parser.parse_args()
    callback(arguments)


if __name__ == "__main__":
    main()
