data_path=$1
experiments="semantic-segmentation/SK_180822_qmciwj41_unet_brain_rtk"
ckpt_path="https://api.blackhole.ai.innopolis.university/pretrained/segmentation_rtk_brain.pt"
if [ -z "$data_path" ]
then
  python infer.py experiments=$experiments \
    "ckpt_path=$ckpt_path"
else
  python infer.py experiments=$experiments \
    "ckpt_path=$ckpt_path" \
    "++datasets.infer.target='$data_path'" \
    "++datasets.infer.source='$data_path'"
fi