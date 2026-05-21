# Ambient Temperature Estimation

A portfolio project implementing a physics-based temperature estimation engine in C++ with Python bindings.

## Overview

This repository contains a custom thermodynamic model and optimizer for estimating device temperature behavior in response to ambient temperature. It combines a numerical ODE solver, derivative-free parameter fitting, and a native Python extension so the model can be trained and evaluated from Python or a Jupyter workflow.

## What it does

- Fits a compact thermal model to time-series temperature data
- Uses a custom ODE describing device temperature dynamics driven by ambient temperature
- Calibrates physical parameters such as heat transfer coefficient `h`, constant input `q`, and initial device temperature `T_dev_0`
- Computes model quality using RMSE over observed vs. simulated temperature traces
- Supports both `train` and `predict` modes through separate model subclasses

## Technology stack

- C++17 for high-performance numerical modeling
- Boost.Odeint for adaptive integration of the thermal ODE
- NLopt (Nelder-Mead) for derivative-free optimization of model parameters
- Native CPython extension using `Python.h` and NumPy C-API
- CMake for native build targets
- Python `setup.py` wrapper for building the extension in-place

## Repository structure

- `src/headers/` — library interfaces and shared data structures
- `src/optimizer/` — fitting, solver, and objective function implementation
- `src/models/` — training/prediction model subclasses and physics definitions
- `src/apis/` — public optimizer API layer
- `src/apis/python/` — Python wrapper and packaging
- `src/test/` — smoke tests for model behavior and training/prediction flow
- `data/` — sample time-series data

## Build instructions

### Native C++ build

```bash
cmake .
make
```

### Python wrapper

```bash
cd src/apis/python
python setup.py build_ext --inplace
```

### Requirements

- Python 3
- NumPy
- NLopt
- Boost (Boost.Odeint)

## Usage

- Run native smoke test targets to validate training and prediction behavior
- Use the Python extension module `ambient_optimizer_python_api` to: initialize the optimizer, feed time-series data, fit the model, and solve/predict with fitted parameters

## Notes

- The current implementation is a single-state physical model with ambient coupling
- The Python wrapper exposes low-level optimizer control and data ingestion via NumPy-compatible arrays
- The CMake targets currently link `libnlopt.dylib` from `/usr/local/lib`, so environment paths may need adjustment

## Credits

Written by [Ben Jordan](https://github.com/bjordan555) and [Andrew Aarestad](https://andrewaarestad.com).
