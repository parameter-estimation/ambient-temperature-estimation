#!/bin/bash

export CFLAGS='-std=c++11'

echo Cleaning with PyCharm venv...

# Install to PyCharm virtualenv
./venv/bin/python setup.py clean
rm -rf build
rm -rf dist
rm -rf optimizer
rm -rf util
rm -rf ambient_optimizer_python_api.egg-info

#python setup.py build_ext --inplace

#python run_predictions.py data/eco_qn3_model_v0.1.0.csv
