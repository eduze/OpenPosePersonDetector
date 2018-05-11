cmake_minimum_required(VERSION 3.5)
project(OpenPersonDetectorAPI)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(OP_INCLUDE_DIRECTORY "" CACHE FILEPATH "Path to OpenPose includes")

if(NOT EXISTS "${OP_INCLUDE_DIRECTORY}/openpose/core/headers.hpp")
    message(SEND_ERROR "Can't find headers.hpp in ${OP_INCLUDE_DIRECTORY}/openpose/core")
endif()

set(PBCVC_INCLUDE_DIRECTORY "" CACHE FILEPATH "Path to pyboostcvconverter includes")

if(NOT EXISTS "${PBCVC_INCLUDE_DIRECTORY}/pyboostcvconverter/pyboostcvconverter.hpp")
    message(SEND_ERROR "Can't find pyboostcvconverter.hpp in ${PBCVC_INCLUDE_DIRECTORY}/pyboostcvconverter")
endif()

set(PYTHON_INCLUDE_DIRECTORY "" CACHE FILEPATH "Path to Python 3 includes")

if(NOT EXISTS "${PYTHON_INCLUDE_DIRECTORY}/Python.h")
    message(SEND_ERROR "Can't find Python.h in ${PYTHON_INCLUDE_DIRECTORY}")
endif()

set(CUDA_HOME_DIRECTORY "" CACHE FILEPATH "Path to CUDA.")

if(NOT EXISTS "${CUDA_HOME_DIRECTORY}/include/cuda.h")
    message(SEND_ERROR "Can't find cuda.h in ${CUDA_HOME_DIRECTORY}/include")
endif()

set(OP_CAFFE_HOME_DIRECTORY "" CACHE FILEPATH "Path to Caffe embedded with OpenPose.")

if(NOT EXISTS "${OP_CAFFE_HOME_DIRECTORY}/include/caffe/caffe.hpp")
    message(SEND_ERROR "Can't find caffe.hpp in ${OP_CAFFE_HOME_DIRECTORY}/include/caffe")
endif()

set(NUMPY_INCLUDE_DIRECTORY "" CACHE FILEPATH "Path to Numpy Includes.")

if(NOT EXISTS "${NUMPY_INCLUDE_DIRECTORY}/numpy/arrayobject.h")
    message(SEND_ERROR "Can't find arrayobject.hpp in ${NUMPY_INCLUDE_DIRECTORY}/numpy")
endif()


set(BOOST_INCLUDE_DIRECTORY "" CACHE FILEPATH "Path to Boost Includes.")

if(NOT EXISTS "${BOOST_INCLUDE_DIRECTORY}/boost/python.hpp")
    message(SEND_ERROR "Can't find python.hpp in ${BOOST_INCLUDE_DIRECTORY}/boost")
endif()


set(PBCVC_LIB_DIRECTORY "" CACHE FILEPATH "Path to libpbcvt.cpython-35m-x86_64-linux-gnup.so file.")

if(NOT EXISTS "${PBCVC_LIB_DIRECTORY}")
    message(SEND_ERROR "Can't find PBCVC lib directory ${PBCVC_LIB_DIRECTORY}")
endif()

set(PYTHON_LIB_DIRECTORY "" CACHE FILEPATH "Path to libpython3.5m.so file.")

if(NOT EXISTS "${PYTHON_LIB_DIRECTORY}/libpython3.5m.so")
    message(SEND_ERROR "Can't find path to libpython3.5m.so ${PYTHON_LIB_DIRECTORY}")
endif()

set(OPENPOSE_LIB_DIRECTORY "" CACHE FILEPATH "Path to libopenpose.so file.")

if(NOT EXISTS "${OPENPOSE_LIB_DIRECTORY}/libopenpose.so")
    message(SEND_ERROR "Can't find path to libopenpose.so ${OPENPOSE_LIB_DIRECTORY}")
endif()

set(OPENCV_LIB_DIRECTORY "" CACHE FILEPATH "Path to opencv lib files.")

if(NOT EXISTS "${OPENCV_LIB_DIRECTORY}/libopencv_core.so")
    message(SEND_ERROR "Can't find ${OPENPOSE_LIB_DIRECTORY}/libopencv_core.so")
endif()

include_directories(${OP_INCLUDE_DIRECTORY} ${PBCVC_INCLUDE_DIRECTORY} ${PYTHON_INCLUDE_DIRECTORY} ${CUDA_HOME_DIRECTORY}/include ${OP_CAFFE_HOME_DIRECTORY}/include ${NUMPY_INCLUDE_DIRECTORY} ${BOOST_INCLUDE_DIRECTORY})
link_directories("${CUDA_HOME_DIRECTORY}/lib64" "${OP_CAFFE_HOME_DIRECTORY}/build/lib/" "${PBCVC_LIB_DIRECTORY}" "${PYTHON_LIB_DIRECTORY}" "${OPENPOSE_LIB_DIRECTORY}" "${OPENCV_LIB_DIRECTORY}")

set(SOURCE_FILES code.cpp)
add_library(OpenPersonDetectorAPI SHARED ${SOURCE_FILES})

target_link_libraries(OpenPersonDetectorAPI "cuda" "cublas" "curand" "caffe" "pbcvt.cpython-35m-x86_64-linux-gnu" "boost_python-py35" "python3.5" "opencv_core" "glog" "boost_python-py35" "gflags" "cudart" "openpose")