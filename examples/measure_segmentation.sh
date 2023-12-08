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
echo "" > nvidiasmi_log
echo "" > mem_log
echo "" > cpu_log

export NO_CLI=True

INITIAL_PROCESSES=$(sudo lsof nohup.out | wc -l) && echo $INITIAL_PROCESSES


nohup time sudo -E env "PATH=$PATH" python train.py experiments=semantic-segmentation/KG_041223_ingieq921_unet.yaml epochs=100 optimizers=adam accelerator=gpu &
PID=$!
echo "Saving cpu+ram info and nvidia-smi to cpu_log + mem_log and nvidiasmi_log"
echo $PID

PROCESSES=$(sudo lsof nohup.out | wc -l) && echo $PROCESSES
while [ $PROCESSES -gt $INITIAL_PROCESSES ]
do
  top -b n1 | grep -E 'Cpu' >> cpu_log
  top -b n1 | grep -E 'MiB Mem' >> mem_log
  sudo nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv | grep "00000000:25:00.0" >> nvidiasmi_log
  sleep 1
  PROCESSES=$(sudo lsof nohup.out | wc -l) # && echo $PROCESSES
done

sudo -E env "PATH=$PATH" python examples/measurements_compaction.py

sudo kill -9 $PID