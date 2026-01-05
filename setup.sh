#!/bin/bash
#conda env create -f environment.yaml
cmake . -B build/ -DCUDAToolkit_ROOT=/usr/local/cuda-12.5 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.5/bin/nvcc
cd build
make
cd ..
cp build/extension/gsrt_cpp_extension.cpython-38-x86_64-linux-gnu.so extension/