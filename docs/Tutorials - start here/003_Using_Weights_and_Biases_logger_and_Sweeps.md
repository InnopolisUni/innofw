# Using the Weights and Biases Logger
[Video](https://www.youtube.com/watch?v=zgt2YFNEOMY&ab_channel=innofw) about using Weights and Biases in the `innofw` framework.

Add logger to the experiment config:

```yaml
# @package _global_
task: "image-segmentation"
defaults:
  - override /models: semantic-segmentation/unet_smp.yaml
  - override /datasets: semantic-segmentation/stroke_dataset.yaml
  - override /losses: semantic-segmentation/focal_tversky_loss.yaml
  - override /loggers: wandb  # <-- add this line to use wandb logger

wandb:  # <-- necessary to add these values
 enable: True
 project: temp_name
 entity: wandb_username
 job_type: training
```

## Sweep
You can get acquainted with what a wandb sweep is in the [documentation](https://docs.wandb.ai/guides/sweeps).

Here we will consider how and where to create configuration files and how to run a sweep in the framework.

1. You need to create a configuration file. Example of a configuration file for a sweep (let's say the file: config/experiments/sweeps/SO_120323_example.yaml):


```yaml
# wandb sweep config
program: train.py
method: bayes
metric:
  name: val_MulticlassJaccardIndex
  goal: maximize

name: param_tuning_deeplabv3plus
project: uw_madison

parameters:

  experiments:
    value: "KG_260423_inwrg21_deeplab_uwmadison.yaml"

  augmentations_train:
    values: [none, 'position/random_horizontal_flip.yaml']

  batch_size:
    values: [ 64, 128, 256 ]

  models.encoder_name:
    values: ['resnet34', 'resnet50', 'resnet101']

  optimizers:
    values: ['lion.yaml', 'adam.yaml']

  optimizers.lr:
    min: 0.0035
    max: 0.0070

  epochs:
    values: [15, 20]

command:
  - ${env}
  - python
  - ${program}
  - ${args_no_hyphens}
```

After this, you need to create a job using the `wandb sweep` command, providing the path to the file. Example of the command:

```bash
wandb sweep config/experiments/sweeps/config/experiments/sweeps/SO_120323_example.yaml
```
Expected command output:

```bash
wandb: Creating sweep from: config/experiments/sweeps/KG_280423_obwg91_sweep_test.yaml
wandb: Created sweep with ID: tx1o222v
wandb: View sweep at: https://wandb.ai/qb-1/uw_madison/sweeps/tx1o222v
wandb: Run sweep agent with: wandb agent qb-1/uw_madison/tx1o222v
```

You need to copy this part: `wandb agent qb-1/uw_madison/tx1o222v`

3. Now you can start the processes to carry out tasks from the job. Here is an example of starting two processes of our task on gpu=0, 2:

```
CUDA_VISIBLE_DEVICES=0 wandb agent qazybi/uwmadison/329fj23
```

```bash
CUDA_VISIBLE_DEVICES=2 wandb agent qazybi/uwmadison/329fj23
```

In case you encounter a `PermissionError`, try running the process in sudo mode:
```bash
CUDA_VISIBLE_DEVICES=2 sudo -E env "PATH=$PATH" wandb agent qazybi/uwmadison/329fj23```
