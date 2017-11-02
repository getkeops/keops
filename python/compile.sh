#!/bin/bash

nvcc -D "USE_DOUBLE_PRECISION=OFF" -D "CUDA_BLOCK_SIZE=128" -Xcompiler -fPIC -shared -o ../build/cuda_conv.so        ../cuda/cuda_conv.cu
nvcc -D "USE_DOUBLE_PRECISION=OFF" -D "CUDA_BLOCK_SIZE=128" -Xcompiler -fPIC -shared -o ../build/cuda_grad1conv.so   ../cuda/cuda_grad1conv.cu
nvcc -D "USE_DOUBLE_PRECISION=OFF" -D "CUDA_BLOCK_SIZE=128" -Xcompiler -fPIC -shared -o ../build/cuda_gradconv_xa.so ../cuda/cuda_gradconv_xa.cu
nvcc -D "USE_DOUBLE_PRECISION=OFF" -D "CUDA_BLOCK_SIZE=128" -Xcompiler -fPIC -shared -o ../build/cuda_gradconv_xx.so ../cuda/cuda_gradconv_xx.cu
nvcc -D "USE_DOUBLE_PRECISION=OFF" -D "CUDA_BLOCK_SIZE=128" -Xcompiler -fPIC -shared -o ../build/cuda_gradconv_xy.so ../cuda/cuda_gradconv_xy.cu
nvcc -D "USE_DOUBLE_PRECISION=OFF" -D "CUDA_BLOCK_SIZE=128" -Xcompiler -fPIC -shared -o ../build/cuda_gradconv_xb.so ../cuda/cuda_gradconv_xb.cu
