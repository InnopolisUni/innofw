pip uninstall opencv-python
pip uninstall opencv-python-headless
pip install opencv-python==3.4.18.65
set CAMERA_URI=rtsp://%CAMERA_IP%/live.sdp
python innofw\onvif_util\stream.py --uri %CAMERA_URI%