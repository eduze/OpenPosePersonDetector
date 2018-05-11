Build content of this directory using CMake and move output file libOpenPersonDetectorAPI.so to project root.

Use CMAKE-GUI to configure parameters of build process.
- Step 1: Download and Compile OpenPose Library. Obtain _[openpose_directory]/build/lib/libopenpose.so_
- Step 2: Navigate to OpenPersonDetectorAPI directory of _CRAMP_Sense_.
- Step 3: Using Terminal, Run `cmake-gui .`
- Step 4: Define a suitable _build directory_. Click Configure.
- Step 5: If prompted, select GNU Compilers as Compiler Suite. 
- Step 6: Provide required paths using UI configuration. Refer tool-tip texts for details.
- Step 7: Click Configure and Generate.
- Step 8: Navigate to _build directory_ and run ```make```.
- Step 9: Copy output file _libOpenPersonDetectorAPI.so_ to Project Root (_CRAMP_Sense_ directory)