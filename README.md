# Open Pose Person Detector

This is a person detector written on top of
[OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose). This
detector is very accuracte compared to commonly used detectors like
**Haar detector**, **HOG Person Detector** and **Tensorflow Object Detection based
detector**.

## Prerequisites

- Compiled version of [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
    - It is recommended to compile caffe shipped with OpenPose as well.
- Boost (`apt install libboost-all-dev`)
- CUDA 8 (Not tested on other versions)
- Numpy
- OpenCV (You will require opencv to be built when compiling OpenPose earlier.
So follow the instructions given by OpenPose)
- Compiled version of [pyboostcvconverter](https://github.com/Algomorph/pyboostcvconverter)

## Installation
Build content of this directory using CMake and move output file
`libOpenPersonDetectorAPI.so` to project root.
Use `CMAKE-GUI` to configure parameters of build process.

- Step 1: Download and Compile OpenPose Library.
Obtain **`openpose_directory`/build/lib/libopenpose.so**

- Step 2: Clone this repo and navigate to root directory of this repo.

- Step 3: Using Terminal, Run `cmake-gui .`.

- Step 4: Define a suitable `build directory`. Click Configure.

- Step 5: If prompted, select GNU Compilers as Compiler Suite.

- Step 6: Provide required paths using UI configuration.
Refer tool-tip texts for details. Usual paths/directories required for CMake:
    - Boost include directory `/usr/include/boost`
    - CUDA home `/usr/local/cuda`
    - Numpy include directory `/usr/include/python3.5m`
    - OpenCV Lib directory `{$OpenCV Build Directory}/lib`
    - OpenPose lib directory `${OpenPose build directory}/src/openpose`
    - OpenPose Caffe Home Directory `${OpenPose project root}/3rdparty/caffe`
    - OpenPose Caffe include Directory `${OpenPose project root}/3rdparty/caffe/include`
    - OpenPose Caffe lib Directory `${OpenPose project root}/3rdparty/caffe/build/lib`
    - OpenPose Include Dir `${OpenPose_root}/include`
    - pyboostcvconverter include directory `${pyboostcvconverter_home}/include`
    - pyboostcvconverter lib directory `${pyboostcvconverter_build_dir}`
    - Python include directory `/usr/include/python3.5`
    - Python Lib Directory `/usr/lib/x86_64-linux-gnu`

- Step 7: Click Configure and Generate.

- Step 8: Navigate to `build directory` and run `make`.

- Step 9: Copy output file `libOpenPersonDetectorAPI.so`
to Project Root of this repo

## Contributions

Contributions are more than welcome including improvements, bug fixes and
adding new issues.