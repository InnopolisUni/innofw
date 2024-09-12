export HYDRA_FULL_ERROR=1
data_path=$1
#ckpt_path="https://api.blackhole.ai.innopolis.university/pretrained/segmentation_unet_brain.pt"
ckpt_path="/home/ainur/data/rtk/weights/sct_april1/1/model.pt"
#ckpt_path="/home/ainur/git/old_innofw/pretrained/segmentation_unet_brain.pt"
experiments="semantic-segmentation/SK_180822_qmciwj41_unet_brain_detection"
if [ -z "$data_path" ]
then
  python infer.py experiments=$experiments \
    "ckpt_path=$ckpt_path"
else
  python infer.py experiments=$experiments \
    "ckpt_path=$ckpt_path" \
    "++datasets.infer.source='$data_path'"
fi