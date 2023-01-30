#!/bin/bash

export CFLAGS='-std=c++11'

echo Building with PyCharm venv...
echo Note: This build script will not pick up header-only changes, you will need to clean in that case

# Install to PyCharm virtualenv
./venv/bin/python build_ext --inplace
./venv/bin/python setup.py install

#python setup.py build_ext --inplace

#python run_predictions.py data/eco_qn3_model_v0.1.0.csv
