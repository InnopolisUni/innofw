yes | pip uninstall opencv-python
yes | pip uninstall opencv-python-headless
sudo apt install libgtk2.0-dev pkg-config cmake -y
yes | pip install opencv-python==3.4.18.65
export CAMERA_URI="rtsp://${CAMERA_IP}/live.sdp"
python innofw/onvif_util/stream.py --uri $CAMERA_URI