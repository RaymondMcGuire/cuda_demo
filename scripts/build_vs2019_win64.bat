cd ..
mkdir build
cd build
cmake .. -G"Visual Studio 16 2019" -A x64 -DCMAKE_CUDA_FLAGS="-arch=sm_61"

pause
