from collections import defaultdict

import cv2
from numpy import max as npmax
from numpy import min as npmin
from numpy import zeros
from sklearn.cluster import DBSCAN


def norming(img):
    mx, mn = npmax(img), npmin(img)
    if mx - mn == 0:
        return img
    return (img - mn) / (mx - mn)


def make_hist(img, how="all", thresh1=10, thresh2=-20):
    temp = defaultdict(int)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            num = int(round(img[i, j]))
            temp[num] += 1
    if how == "ess":
        black = min(temp.keys())
        for k in temp.keys():
            if not (temp[k] > thresh1 and k > max(black, thresh2)):
                del temp[k]
    return temp


def make_kernel_trick(img, how="all"):
    answer = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if how == "all" or (how == "nonzero" and img[i, j] > 0):
                answer.append((img[i, j], i, j))
    return answer


def dekernel(zipped, shape=(512, 512)):
    img = zeros(shape)
    for x, y in zipped:
        try:
            img[x[1], x[2]] = y
        except:
            if (x[1] < 0) or (x[1] > shape[0]):
                print("x[1] is out of bounds [0,", shape[0], "]")
            if (x[2] < 0) or (x[2] > shape[1]):
                print("x[2] is out of bounds [0,", shape[1], "]")
    return img


def make_mask(img, cluster=0):
    newimg = zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] == cluster:
                newimg[i, j] = 1
            else:
                newimg[i, j] = 0
    return newimg


def make_contrasted(img, contrast=20):
    # these values have been found experimentally and work for brain CT images.
    newimg = img
    f = 131 * (contrast + 127) / (127 * (131 - contrast))
    newimg = f * (newimg - 127) + 127
    return newimg


class MakeContrasted:
    """
    Custom preprocessing pipline to make some parts of the images more contrasted
    ...

    Attributes
    ----------

    Methods
    -------
    """

    def __call__(
        self,
        image,
        bbox=None,
        mask=None,
        keypoints=None,
        force_apply=False,
        *args,
        **kwargs,
    ):
        kerneled = make_kernel_trick(image)
        model = DBSCAN(eps=5, min_samples=1, n_jobs=4)
        model.fit(kerneled)
        dekerneled = dekernel(zip(kerneled, model.labels_), image.shape)
        mask = make_mask(dekerneled)
        contrasted = mask * image
        contrasted = make_contrasted(contrasted)

        return {
            "image": contrasted,
            "bbox": bbox,
            "mask": mask,
            "keypoints": keypoints,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process input/output image paths"
    )
    parser.add_argument(
        "-i",
        "--input_path",
        required=True,
        help="path to lung image in png format",
    )
    parser.add_argument(
        "-o", "--output_path", required=True, help="path to save image"
    )
    argv = parser.parse_args()
    img = cv2.imread(argv.input_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contraster = MakeContrasted()
    contrasted = contraster(img)["image"]
    cv2.imwrite(argv.output_path, contrasted)
