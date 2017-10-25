#!/bin/bash

optirun nvcc -D "USE_DOUBLE_PRECISION=OFF" -D "CUDA_BLOCK_SIZE=128" -Xcompiler -fPIC -shared -o ./build/cuda_conv.so        ./cuda/cuda_conv.cu
optirun nvcc -D "USE_DOUBLE_PRECISION=OFF" -D "CUDA_BLOCK_SIZE=128" -Xcompiler -fPIC -shared -o ./build/cuda_grad1conv.so   ./cuda/cuda_grad1conv.cu
optirun nvcc -D "USE_DOUBLE_PRECISION=OFF" -D "CUDA_BLOCK_SIZE=128" -Xcompiler -fPIC -shared -o ./build/cuda_gradconv_xa.so ./cuda/cuda_gradconv_xa.cu
optirun nvcc -D "USE_DOUBLE_PRECISION=OFF" -D "CUDA_BLOCK_SIZE=128" -Xcompiler -fPIC -shared -o ./build/cuda_gradconv_xx.so ./cuda/cuda_gradconv_xx.cu
optirun nvcc -D "USE_DOUBLE_PRECISION=OFF" -D "CUDA_BLOCK_SIZE=128" -Xcompiler -fPIC -shared -o ./build/cuda_gradconv_xy.so ./cuda/cuda_gradconv_xy.cu
optirun nvcc -D "USE_DOUBLE_PRECISION=OFF" -D "CUDA_BLOCK_SIZE=128" -Xcompiler -fPIC -shared -o ./build/cuda_gradconv_xb.so ./cuda/cuda_gradconv_xb.cu
