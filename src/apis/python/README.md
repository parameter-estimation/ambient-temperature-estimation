# Optimizer Python API

CPython extension that wraps the C++ Optimizer library. The C++ core is designed for edge deployment; this binding is the lab-side counterpart for calibration, analysis, and notebook-driven experimentation against the exact same model code that runs on-target.

## Build

```bash
python setup.py build_ext --inplace
```

This produces a `.so` (or `.dll`) in place. The build links against NLopt and Boost and compiles the C++ sources from `../optimizer/` and `../optimizer_api.cpp` directly into the extension — no separate library install step.

Requires Python 3.8 or 3.9 (the build uses `distutils`, removed in Python 3.12), NumPy, NLopt, and Boost.

## Usage

```python
import ambient_optimizer_python_api as aopa
```

The module exposes five functions: `init`, `feed`, `fit`, `generate`, `version`. See the top-level [README](../../../README.md#python-api) for the call signatures, and `test.py` / `examples/*.ipynb` for worked examples.

## Notes

- `feed` and the NumPy interop currently copy element-by-element rather than sharing buffers — fine for calibration workloads, would want revisiting if pushed at high frequency.
- The example `prediction.ipynb` references a CSV path (`../data/cr_20/csv/...`) that is not checked into this repo; point it at `data/long_chamber_data.csv` or your own logged trace.
