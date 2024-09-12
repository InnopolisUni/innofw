#!/bin/bash
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
