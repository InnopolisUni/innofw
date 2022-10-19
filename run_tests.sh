#!/bin/bash
export PYTHONPATH=.
python -m pytest --cov=innofw --cov-report=xml  --junitxml=/code/out_report.xml
# coverage report --fail-under=40
if [ $? -ne 0 ]; then
        echo ERROR
        exit 2
else
        echo OK
fi
