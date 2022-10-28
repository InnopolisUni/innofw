#!/bin/bash
PWD=$(pwd)
if [ "${PWD##*/}"="examples" ]; then
  cd ..
fi
nohup python train.py experiments=IM_190722_vwer3f23_oneshotlearning.yaml  &
PID=$!
sudo iotop -bot -d 0.1 | grep $PID
kill $PID