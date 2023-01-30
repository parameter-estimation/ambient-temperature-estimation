from distutils.core import setup, Extension
import numpy

extra_objects = ['/usr/local/lib/libnlopt.dylib']
libraries = ['nlopt']

c_ext = Extension(
    "ambient_optimizer_python_api",
    sources=[
        "python_wrapper.cpp",
        "../optimizer_api.cpp",
        "../../optimizer/optimizer_data_buffer.cpp",
        "../../optimizer/optimizer_objective_functions.cpp",
        "../../optimizer/optimizer_fitting.cpp",
        "../../optimizer/optimizer_solver.cpp",
        "../../util/utils.cpp"

    ],
    libraries=libraries,
    extra_objects=extra_objects,
    extra_compile_args=[
        "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
        "-std=c++17"
    ]
)

setup(
    name='ambient_optimizer_python_api',
    version='1.0.0',
    ext_modules=[c_ext],
    include_dirs=[
        "../../headers",
        "../../models",
        numpy.get_include()
    ]
)
