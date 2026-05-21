
# Optimizer Python API

This Python module is a native extension that wraps the C++ Optimizer library, enabling direct use of high-performance C++ code from Python. This approach allows the same core model and optimization logic to be used in both embedded/edge deployments (C++) and flexible lab or research environments (Python/Jupyter), ensuring consistency and reducing code duplication.

## Build Instructions

To build the library from the source C++ files, run the following command from a terminal:

```bash
python setup.py build_ext --inplace
```

This will rebuild the `.so` or `.dll` file, depending on your platform. The build links against NLopt and Boost, and uses the NumPy C-API for efficient data transfer.

## Usage

Import the module in your Python code:

```python
import ambient_optimizer_python_api as aopa
```

You can then initialize the optimizer, feed time-series data, fit the model, and run predictions using the same C++ code that runs on-device. This enables rapid prototyping, interactive analysis, and reproducible research, all while guaranteeing that production and research environments use identical model logic.

