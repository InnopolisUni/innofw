#!/bin/bash
PWD=$(pwd)

current_dir=${PWD##*/}

var2="examples"
if [ "$current_dir" = "$var2" ]; then
  cd ..
fi
nohup python train.py experiments=one-shot-learning/IM_190722_vwer3f23_oneshotlearning.yaml  &
PID=$!
sudo iotop -bot -d 0.1 | grep $PID
kill $PID