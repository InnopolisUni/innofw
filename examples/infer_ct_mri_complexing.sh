export HYDRA_FULL_ERROR=1
data_path=$1
#ckpt_path="https://api.blackhole.ai.innopolis.university/pretrained/segmentation_unet_brain_comples.pt"
experiments="semantic-segmentation/SK_100923_unet_brain_mri.yaml"
if [ -z "$data_path" ]
then
  python infer.py experiments=$experiments \
    "ckpt_path=$ckpt_path"
else
  echo $data_path
  python infer.py experiments=$experiments \
    "++datasets.infer.source='$data_path'"
#    "ckpt_path=$ckpt_path" \
fi
