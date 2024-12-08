data_path=$1
ckpt_path="https://api.blackhole.ai.innopolis.university/pretrained/segmentation_rtk_brain.pt"
experiments="semantic-segmentation/AK_081224_Yjc97FX_unet_brain_complex.yaml"

if [ -z "$data_path" ]
then
  python infer.py experiments=$experiments \
    "ckpt_path=$ckpt_path"
else
  python infer.py experiments=$experiments \
    "++datasets.infer.source='$data_path'" \
    "ckpt_path=$ckpt_path"
fi
