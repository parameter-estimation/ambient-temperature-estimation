cmake_minimum_required(VERSION 3.0)
project (Optimizer)

set(CMAKE_CXX_STANDARD 17)

include_directories(headers)
include_directories(models)
include_directories(SYSTEM /usr/local/include)

# Grab Python, 3.8 or newer
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

#message(STATUS "Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
#message(STATUS "Python3_NumPy_INCLUDE_DIRS: ${Python3_NumPy_INCLUDE_DIRS}")

add_executable(PythonBindingsTest
        test/test_python_wrapper.cpp
        apis/python/python_wrapper.cpp
        optimizer/optimizer_data_buffer.cpp
        optimizer/optimizer_objective_functions.cpp
        apis/optimizer_api.cpp
        optimizer/optimizer_fitting.cpp
        optimizer/optimizer_solver.cpp
        util/utils.cpp)

target_include_directories(PythonBindingsTest PUBLIC ${Python3_INCLUDE_DIRS})
target_include_directories(PythonBindingsTest PUBLIC ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(PythonBindingsTest ${Python3_LIBRARIES})
target_link_libraries(PythonBindingsTest /usr/local/lib/libnlopt.dylib)

add_executable(SmokeTestTrain
        test/smoke_test_train.cpp
        optimizer/optimizer_data_buffer.cpp
        optimizer/optimizer_objective_functions.cpp
        apis/optimizer_api.cpp
        optimizer/optimizer_fitting.cpp
        optimizer/optimizer_solver.cpp
        util/utils.cpp)

# you must have nlopt installed in /usr/local/lib
target_link_libraries(SmokeTestTrain /usr/local/lib/libnlopt.dylib)

add_executable(SmokeTestPredict
        test/smoke_test_predict.cpp
        optimizer/optimizer_data_buffer.cpp
        optimizer/optimizer_objective_functions.cpp
        apis/optimizer_api.cpp
        optimizer/optimizer_fitting.cpp
        optimizer/optimizer_solver.cpp
        util/utils.cpp)

# you must have nlopt installed in /usr/local/lib
target_link_libraries(SmokeTestPredict /usr/local/lib/libnlopt.dylib)