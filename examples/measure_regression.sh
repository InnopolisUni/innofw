#!/bin/bash
# run like this: sudo -E env "PATH=$PATH" bash examples/measure_image_classification.sh
sudo apt-get install -y iotop > /dev/null
PWD=$(pwd)

current_dir=${PWD##*/}

var2="examples"
if [ "$current_dir" = "$var2" ]; then
  cd ..
fi
echo "" > nohup.out
echo "" > mem_log
echo "" > cpu_log

export NO_CLI=True

nohup time sudo -E env "PATH=$PATH" python train.py experiments=regression/KA_190722_0a70ef39_xgbregressor_cc.yaml &
PID=$!
echo "Saving cpu+ram info to cpu_log+mem_log"
echo $PID

while true
do 
  top -b n1 | grep -E 'Cpu' >> cpu_log
  top -b n1 | grep -E 'MiB Mem' >> mem_log
  sleep 1
done 

kill $PID