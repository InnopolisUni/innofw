#!/bin/bash
sudo apt-get install -y iotop > /dev/null
PWD=$(pwd)

current_dir=${PWD##*/}

var2="examples"
if [ "$current_dir" = "$var2" ]; then
  cd ..
fi
nohup python train.py experiments=KA_130722_9f7134db_linear_regression.yaml &
PID=$!
sudo iotop -bot -d 0.1 | grep $PID
kill $PID