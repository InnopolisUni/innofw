# Использование логгера Weights and Biases

[Видео](https://www.youtube.com/watch?v=zgt2YFNEOMY&ab_channel=innofw) про использование Weights and Biases во фреймворке `innofw`.

добавить логгер в конфиг эксперимента

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

с тем что такое wandb sweep можно ознакомиться по [документации](https://docs.wandb.ai/guides/sweeps)

Здесь же мы рассмотрим как и где создать конфигурационные файлы и как запустить sweep во фреймворке.

1. Нужно создать конфигурационный файл. Пример конфигурационного файла для sweep(допустим файл: `config/experiments/sweeps/SO_120323_example.yaml`):
    
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
    
2. После этого надо будет создать job командой `wandb sweep` передав путь до файла. Пример команды:
    
    ```bash
    wandb sweep config/experiments/sweeps/config/experiments/sweeps/SO_120323_example.yaml
    ```
    
    Примерный вывод команды:
    
    ```bash
    wandb: Creating sweep from: config/experiments/sweeps/KG_280423_obwg91_sweep_test.yaml
    wandb: Created sweep with ID: tx1o222v
    wandb: View sweep at: https://wandb.ai/qb-1/uw_madison/sweeps/tx1o222v
    wandb: Run sweep agent with: wandb agent qb-1/uw_madison/tx1o222v
    ```
    
    Нужно скопировать часть: `wandb agent qb-1/uw_madison/tx1o222v`
    
3. Теперь можно запустить процессы, чтобы выполнялись задачи с job-а. Вот пример запуска двух процессов нашей задачи на гпу=0, 2: 
    
    ```bash
    CUDA_VISIBLE_DEVICES=0 wandb agent qazybi/uwmadison/329fj23
    ```
    
    ```bash
    CUDA_VISIBLE_DEVICES=2 wandb agent qazybi/uwmadison/329fj23
    ```
    
    В случае если возникает ошибка `PermissionError`, попробуйте запустить процесс в режиме sudo:
    ```bash
    CUDA_VISIBLE_DEVICES=2 sudo -E env "PATH=$PATH" wandb agent qazybi/uwmadison/329fj23
    ```
