
# Ambient Temperature Estimation

A portfolio project implementing a physics-based temperature estimation engine in C++ with robust Python bindings.

## Overview

This repository demonstrates a cross-environment approach to scientific modeling: the core thermodynamic model and optimizer are written in C++ for maximum performance and portability, while a native Python extension exposes the same codebase for interactive analysis, rapid prototyping, and lab automation. This enables seamless reuse of the exact same C++ logic in both embedded/edge deployments and flexible Python-driven research workflows (e.g., Jupyter, data science pipelines).

The project combines a custom ODE solver, derivative-free parameter fitting, and a CPython/NumPy extension so the model can be trained and evaluated from Python or C++ with no code duplication.


## What it does

- Fits a compact thermal model to time-series temperature data
- Uses a custom ODE describing device temperature dynamics driven by ambient temperature
- Calibrates physical parameters such as heat transfer coefficient `h`, constant input `q`, and initial device temperature `T_dev_0`
- Computes model quality using RMSE over observed vs. simulated temperature traces
- Supports both `train` and `predict` modes through separate model subclasses
- Enables the same C++ code to be used in both edge (embedded) and lab (Python) environments, reducing maintenance and ensuring consistency


## Technology stack

- C++17 for high-performance numerical modeling
- Boost.Odeint for adaptive integration of the thermal ODE
- NLopt (Nelder-Mead) for derivative-free optimization of model parameters
- Native CPython extension using `Python.h` and NumPy C-API for zero-copy data transfer
- CMake for native build targets
- Python `setup.py` wrapper for building the extension in-place


## Repository structure

- `src/headers/` — library interfaces and shared data structures
- `src/optimizer/` — fitting, solver, and objective function implementation
- `src/models/` — training/prediction model subclasses and physics definitions
- `src/apis/` — public optimizer API layer
- `src/apis/python/` — Python wrapper and packaging (enables cross-environment use)
- `src/test/` — smoke tests for model behavior and training/prediction flow
- `data/` — sample time-series data


## Build instructions

### Native C++ build

```bash
cmake .
make
```

### Python wrapper (cross-environment)

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
- In both edge and lab environments, the same C++ code is used for model logic, ensuring results are consistent and reproducible across deployment targets


## Notes

- The current implementation is a single-state physical model with ambient coupling
- The Python wrapper exposes low-level optimizer control and data ingestion via NumPy-compatible arrays
- The CMake targets currently link `libnlopt.dylib` from `/usr/local/lib`, so environment paths may need adjustment
- This project demonstrates a best-practice pattern for scientific/engineering code: write core logic in C++ for portability and performance, then wrap with Python for usability and rapid iteration

## Credits

Written by [Ben Jordan](https://github.com/bjordan555) and [Andrew Aarestad](https://andrewaarestad.com).
