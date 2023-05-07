import pytest
import torch


def test_training_with_checkpoint(
    fitted_segmentation_module,
    trainer_with_temporary_directory,
    dummy_data_module,
):
    trainer, checkpoint_dir = trainer_with_temporary_directory
    dataloader = dummy_data_module.train_dataloader()

    # First training phase is already done in the fitted_segmentation_module fixture
    last_checkpoint_path = (
        fitted_segmentation_module.trainer.checkpoint_callback.best_model_path
    )

    # Continue training using the fitted_segmentation_module
    trainer.fit(
        fitted_segmentation_module,
        ckpt_path=last_checkpoint_path,
        train_dataloaders=dataloader,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="No GPU is found on this machine"
)
def test_training_without_checkpoint_with_gpu(
    segmentation_module, trainer_with_temporary_directory, dummy_data_module
):
    trainer, checkpoint_dir = trainer_with_temporary_directory
    dataloader = dummy_data_module.train_dataloader()
    trainer.fit(segmentation_module, train_dataloaders=dataloader)


def test_testing_without_checkpoint(
    fitted_segmentation_module, dummy_data_module
):
    trainer = fitted_segmentation_module.trainer
    dataloader = dummy_data_module.test_dataloader()
    test_results = trainer.test(
        fitted_segmentation_module, dataloaders=dataloader
    )

    # Check if test_results is a non-empty list
    assert len(test_results) > 0

    # Check if the first dictionary in test_results contains the expected metrics
    expected_metrics = {
        "test_BinaryF1Score",
        "test_BinaryJaccardIndex",
        "test_BinaryPrecision",
        "test_BinaryRecall",
    }

    training_metric_values = fitted_segmentation_module.training_metric_values
    metric_name = "BinaryF1Score"

    for i in range(1, len(training_metric_values)):
        assert (
            training_metric_values[i][metric_name]
            > training_metric_values[i - 1][metric_name]
        )


def test_predicting_without_checkpoint(
    fitted_segmentation_module, dummy_data_module
):
    trainer = fitted_segmentation_module.trainer
    dataloader = dummy_data_module.train_dataloader()
    trainer.predict(fitted_segmentation_module, dataloaders=dataloader)


def test_testing_with_checkpoint(
    fitted_segmentation_module, dummy_data_module
):
    trainer = fitted_segmentation_module.trainer
    dataloader = dummy_data_module.train_dataloader()
    last_checkpoint_path = trainer.checkpoint_callback.best_model_path

    # Test the loaded model
    trainer.test(
        fitted_segmentation_module,
        ckpt_path=last_checkpoint_path,
        dataloaders=dataloader,
    )


def test_predicting_with_checkpoint(
    fitted_segmentation_module, dummy_data_module
):
    trainer = fitted_segmentation_module.trainer
    dataloader = dummy_data_module.train_dataloader()
    last_checkpoint_path = trainer.checkpoint_callback.best_model_path

    # Create a DataLoader for the prediction data
    trainer.predict(
        fitted_segmentation_module,
        ckpt_path=last_checkpoint_path,
        dataloaders=dataloader,
    )
