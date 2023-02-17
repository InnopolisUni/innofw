from innofw.core.datasets.semantic_segmentation.tiff_dataset import SegmentationDataset


# for now just create a folder with segmentation dataset
# later I will upload it to the minio

# I should have a config file for the dataset

# dataset should contain 4 images and 4 masks

# todo: add full iteration over the dataset

def test_read(images, masks, transform, channels, with_caching):
    ds = SegmentationDataset(
        images,
        masks,
        transform,
        channels,
        with_caching,
    )


def test_with_caching():
    pass


def test_with_transform():
    pass


def test_with_transform_n_caching():
    pass


def test_with_mask():
    pass


def test_with_mask_n_caching():
    pass


def test_with_mask_n_transform():
    pass


def test_with_mask_n_transform_n_caching():
    pass


def test_wrong_img_mask_number():
    pass


def test_wrong_img():
    pass


def test_channels():
    # data should contain 4 channels
    # but 3 needed
    pass
