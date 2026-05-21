cmake_minimum_required(VERSION 3.0)
project (Optimizer)

set(CMAKE_CXX_STANDARD 17)

include_directories(src/headers)
include_directories(src/models)
include_directories(SYSTEM /usr/local/include)

# Grab Python, 3.8 or newer
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

#message(STATUS "Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
#message(STATUS "Python3_NumPy_INCLUDE_DIRS: ${Python3_NumPy_INCLUDE_DIRS}")

add_executable(PythonBindingsTest
#        src/test/test_python_wrapper.cpp
        src/apis/python/python_wrapper.cpp
        src/optimizer/optimizer_data_buffer.cpp
        src/optimizer/optimizer_objective_functions.cpp
        src/apis/optimizer_api.cpp
        src/optimizer/optimizer_fitting.cpp
        src/optimizer/optimizer_solver.cpp
        src/util/utils.cpp)

target_include_directories(PythonBindingsTest PUBLIC ${Python3_INCLUDE_DIRS})
target_include_directories(PythonBindingsTest PUBLIC ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(PythonBindingsTest ${Python3_LIBRARIES})
target_link_libraries(PythonBindingsTest /usr/local/lib/libnlopt.dylib)

add_executable(SmokeTestTrain
        src/test/smoke_test_train.cpp
        src/optimizer/optimizer_data_buffer.cpp
        src/optimizer/optimizer_objective_functions.cpp
        src/apis/optimizer_api.cpp
        src/optimizer/optimizer_fitting.cpp
        src/optimizer/optimizer_solver.cpp
        src/util/utils.cpp)

# you must have nlopt installed in /usr/local/lib
target_link_libraries(SmokeTestTrain /usr/local/lib/libnlopt.dylib)

add_executable(SmokeTestPredict
        src/test/smoke_test_predict.cpp
        src/optimizer/optimizer_data_buffer.cpp
        src/optimizer/optimizer_objective_functions.cpp
        src/apis/optimizer_api.cpp
        src/optimizer/optimizer_fitting.cpp
        src/optimizer/optimizer_solver.cpp
        src/util/utils.cpp)

# you must have nlopt installed in /usr/local/lib
target_link_libraries(SmokeTestPredict /usr/local/lib/libnlopt.dylib)