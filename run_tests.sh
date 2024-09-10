#!/bin/bash
python -m pip install -U openmim
python -m pip install -U open3d
mim install mmengine
mim install 'mmcv>=2.0.0rc4'
mim install 'mmdet>=3.0.0'
export PYTHONPATH=.
python -m pytest --cov=innofw --cov-report=xml  --junitxml=out_report.xml --cov-fail-under=10 # not percents
if [ $? -ne 0 ]; then
  echo ERROR
  exit 2
fi
coverage report --fail-under=50 # percents
if [ $? -ne 0 ]; then
        echo ERROR
        exit 2
else
        echo OK
fi
