#!/bin/bash
sudo add-apt-repository ppa:flexiondotorg/nvtop -y  #  > /dev/null
sudo apt install nvtop  #  > /dev/null
PWD=$(pwd)
if [ "${PWD##*/}"="examples" ]; then
  cd ..
fi
nohup python train.py experiments=KA_230922_sdgh32lk_unet.yaml ++trainer.accelerator=gpu &
PID=$!
sudo nvtop
 #-bot -d 0.1 | grep $PID
kill $PID