start procexp.exe &
python train.py experiments=IM_190722_vwer3f23_oneshotlearning.yaml optimizers=auto accelerator=gpu +devices=1 +batch_size=1 datasets.num_workers=2
