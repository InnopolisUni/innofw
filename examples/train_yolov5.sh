ckpt_path=https://api.blackhole.ai.innopolis.university/pretrained/lep_yolov5.pt
#ckpt_path=logs/train/lep/KA_120722_8adfcdaa_yolov5/20221010-0709542/weights/last5.pt
python train.py experiments=KA_120722_8adfcdaa_yolov5  +ckpt_path=$ckpt_path
