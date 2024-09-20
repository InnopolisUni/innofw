import os

import pytest
import torch


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="No GPU is found on this machine"
)
def test_training_with_gpu(
    segmentation_module_function_scope,
    trainer_with_temporary_directory,
    seg_dummy_data_module,
):
    trainer, checkpoint_dir = trainer_with_temporary_directory
    dataloader = seg_dummy_data_module.train_dataloader()
    trainer.fit(segmentation_module_function_scope, train_dataloaders=dataloader)


def test_training_without_checkpoint(
    segmentation_module_function_scope,
    trainer_with_temporary_directory,
    seg_dummy_data_module,
):
    trainer, checkpoint_dir = trainer_with_temporary_directory
    dataloader = seg_dummy_data_module.train_dataloader()
    trainer.fit(segmentation_module_function_scope, train_dataloaders=dataloader)


def test_training_with_checkpoint(
    fitted_segmentation_module,
    trainer_with_temporary_directory,
    seg_dummy_data_module,
):
    trainer, checkpoint_dir = trainer_with_temporary_directory
    dataloader = seg_dummy_data_module.train_dataloader()

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


def test_testing_without_checkpoint(
    segmentation_module_function_scope,
    seg_dummy_data_module,
    trainer_with_temporary_directory,
):
    trainer, checkpoint_dir = trainer_with_temporary_directory
    dataloader = seg_dummy_data_module.train_dataloader()

    # Test the loaded model
    trainer.test(
        segmentation_module_function_scope,
        dataloaders=dataloader,
    )


def test_testing_with_checkpoint(
    segmentation_module_function_scope,
    fitted_segmentation_module,
    seg_dummy_data_module,
    trainer_with_temporary_directory,
):
    fitted_model_trainer = fitted_segmentation_module.trainer
    fitted_model_dataloader = seg_dummy_data_module.test_dataloader()

    # Get the checkpoint directory and list all checkpoint files
    checkpoint_dir = fitted_model_trainer.checkpoint_callback.dirpath
    checkpoints = sorted(os.listdir(checkpoint_dir))
    first_checkpoint_path = os.path.join(checkpoint_dir, checkpoints[0])
    last_checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])

    # Test with the first checkpoint
    first_checkpoint_test_results = fitted_model_trainer.test(
        fitted_segmentation_module,
        dataloaders=fitted_model_dataloader,
        ckpt_path=first_checkpoint_path,
    )

    # Test with the last checkpoint
    last_checkpoint_test_results = fitted_model_trainer.test(
        fitted_segmentation_module,
        dataloaders=fitted_model_dataloader,
        ckpt_path=last_checkpoint_path,
    )

    for key in last_checkpoint_test_results[0].keys():
        assert (
            last_checkpoint_test_results[0][key]
            > 0  # first_checkpoint_test_results[0][key]
        )


def test_predicting_without_checkpoint(
    segmentation_module_function_scope,
    seg_dummy_data_module,
    trainer_with_temporary_directory,
):
    trainer, checkpoint_dir = trainer_with_temporary_directory
    fitted_model_dataloader = seg_dummy_data_module.test_dataloader()
    trainer.predict(
        segmentation_module_function_scope, dataloaders=fitted_model_dataloader
    )


def test_predicting_with_checkpoint(fitted_segmentation_module, seg_dummy_data_module):
    trainer = fitted_segmentation_module.trainer
    dataloader = seg_dummy_data_module.train_dataloader()
    last_checkpoint_path = trainer.checkpoint_callback.best_model_path

    # Create a DataLoader for the prediction data
    trainer.predict(
        fitted_segmentation_module,
        ckpt_path=last_checkpoint_path,
        dataloaders=dataloader,
    )
