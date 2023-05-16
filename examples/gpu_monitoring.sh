#!/bin/bash
export NO_CLI=True
#sudo add-apt-repository ppa:flexiondotorg/nvtop -y > /dev/null
#sudo apt install nvtop > /dev/null

sudo apt install libdrm-dev libsystemd-dev
sudo apt install libudev-dev
sudo apt install cmake libncurses5-dev libncursesw5-dev git

git clone https://github.com/Syllo/nvtop.git
mkdir -p nvtop/build && cd nvtop/build
cmake .. -DNVIDIA_SUPPORT=ON -DAMDGPU_SUPPORT=ON -DINTEL_SUPPORT=ON
make

# Install globally on the system
sudo make install

cd ../..
sudo rm -r nvtop


nohup python train.py experiments=one-shot-learning/IM_190722_vwer3f23_oneshotlearning.yaml optimizers=adam accelerator=gpu +devices=1 +batch_size=128 &
PID=$!
sudo nvtop
kill $PID
