#!/bin/bash

# --------- # PATH: --------- #

CUDAROOT="/usr/local/cuda"
NVCC="/usr/local/cuda/bin/nvcc"
CC="/usr/bin/gcc"
LIBKPROOT=".."

INSTALL_DIR="$LIBKPROOT/python/pykp/build"

# --------- # GPU PARAMETERS: --------- #

COMPUTECAPABILITY=35
TYPE=float
BLOCKSIZE=192

# --------- # NVCC PARAMETERS: --------- #

NVCCFLAGS="-ccbin=$CC -arch=sm_$COMPUTECAPABILITY -Xcompiler -fPIC -shared"
LIBKPPATH_CONV="$LIBKPROOT/cuda/specific/radial_kernels"
LIBKPPATH_DIST="$LIBKPROOT/cuda/specific/shape_distance"

# --------- FSHAPES DISTANCES PARAMETERS: --------- #
KGEOM=("gaussian" "cauchy")
KSIG=("gaussian" "cauchy")
KSPHERE=("gaussian_unoriented" "binet" "gaussian_oriented" "linear")

KERNEL_GEOM=(0 1)
KERNEL_SIG=(0 1)
KERNEL_SPHERE=(0 1 2 3)

#clean 
rm -f $INSTALL_DIR/*.so

#create shared object with nvcc
echo "Compiling $LIBKPPATH_CONV/cuda_conv.cu..."
$NVCC -D "__TYPE__=$TYPE" -D "CUDA_BLOCK_SIZE=$BLOCKSIZE" $NVCCFLAGS -I$LIBKPROOT/cuda -I$LIBKPPATH_CONV -o "$INSTALL_DIR/cuda_conv.so"        "$LIBKPPATH_CONV/cuda_conv.cu"
echo "Compiling $LIBKPPATH_CONV/cuda_grad1conv.cu..."
$NVCC -D "__TYPE__=$TYPE" -D "CUDA_BLOCK_SIZE=$BLOCKSIZE" $NVCCFLAGS -I$LIBKPROOT/cuda -I$LIBKPPATH_CONV -o "$INSTALL_DIR/cuda_grad1conv.so"   "$LIBKPPATH_CONV/cuda_grad1conv.cu"
echo "Compiling $LIBKPPATH_CONV/cuda_gradconv_xa.cu..."
$NVCC -D "__TYPE__=$TYPE" -D "CUDA_BLOCK_SIZE=$BLOCKSIZE" $NVCCFLAGS -I$LIBKPROOT/cuda -I$LIBKPPATH_CONV -o "$INSTALL_DIR/cuda_gradconv_xa.so" "$LIBKPPATH_CONV/cuda_gradconv_xa.cu"
echo "Compiling $LIBKPPATH_CONV/cuda_gradconv_xx.cu..."
$NVCC -D "__TYPE__=$TYPE" -D "CUDA_BLOCK_SIZE=$BLOCKSIZE" $NVCCFLAGS -I$LIBKPROOT/cuda -I$LIBKPPATH_CONV -o "$INSTALL_DIR/cuda_gradconv_xx.so" "$LIBKPPATH_CONV/cuda_gradconv_xx.cu"
echo "Compiling $LIBKPPATH_CONV/cuda_gradconv_xy.cu..."
$NVCC -D "__TYPE__=$TYPE" -D "CUDA_BLOCK_SIZE=$BLOCKSIZE" $NVCCFLAGS -I$LIBKPROOT/cuda -I$LIBKPPATH_CONV -o "$INSTALL_DIR/cuda_gradconv_xy.so" "$LIBKPPATH_CONV/cuda_gradconv_xy.cu"
echo "Compiling $LIBKPPATH_CONV/cuda_gradconv_xb.cu..."
$NVCC -D "__TYPE__=$TYPE" -D "CUDA_BLOCK_SIZE=$BLOCKSIZE" $NVCCFLAGS -I$LIBKPROOT/cuda -I$LIBKPPATH_CONV -o "$INSTALL_DIR/cuda_gradconv_xb.so" "$LIBKPPATH_CONV/cuda_gradconv_xb.cu"

for i in ${KERNEL_GEOM[@]}; do 
    for j in ${KERNEL_SIG[@]}; do
        for k in ${KERNEL_SPHERE[@]}; do
            echo "Compiling $LIBKPPATH_DIST/fshape_gpu.cu with kernel geom: ${KGEOM[$i]}, kernel sig: ${KSIG[$j]}, kernel sphere: ${KSPHERE[$k]}..."
            $NVCC -D "__TYPE__=$TYPE" -D "CUDA_BLOCK_SIZE=$BLOCKSIZE" -D "KERNEL_GEOM_TYPE=$i" -D "KERNEL_SIG_TYPE=$j" -D "KERNEL_SPHERE_TYPE=$k" $NVCCFLAGS -I$LIBKPPATH_DIST -o "$INSTALL_DIR/cuda_fshape_scp_${KGEOM[$i]}${KSIG[$j]}${KSPHERE[$k]}.so" "$LIBKPPATH_DIST/fshape_gpu.cu";
            echo "Compiling $LIBKPPATH_DIST/fshape_gpu_dx.cu with kernel geom: ${KGEOM[$i]}, kernel sig: ${KSIG[$j]}, kernel sphere: ${KSPHERE[$k]}..."
            $NVCC -D "__TYPE__=$TYPE" -D "CUDA_BLOCK_SIZE=$BLOCKSIZE" -D "KERNEL_GEOM_TYPE=$i" -D "KERNEL_SIG_TYPE=$j" -D "KERNEL_SPHERE_TYPE=$k" $NVCCFLAGS -I$LIBKPPATH_DIST -o "$INSTALL_DIR/cuda_fshape_scp_dx_${KGEOM[$i]}${KSIG[$j]}${KSPHERE[$k]}.so" "$LIBKPPATH_DIST/fshape_gpu_dx.cu";
            #echo "Compiling $LIBKPPATH_DIST/fshape_gpu_dxi.cu with kernel geom: ${KGEOM[$i]}, kernel sig: ${KSIG[$j]}, kernel sphere: ${KSPHERE[$k]}..."
            #$NVCC -D "__TYPE__=$TYPE" -D "CUDA_BLOCK_SIZE=$BLOCKSIZE" -D "KERNEL_GEOM_TYPE=$i" -D "KERNEL_SIG_TYPE=$j" -D "KERNEL_SPHERE_TYPE=$k" $NVCCFLAGS -I$LIBKPPATH_DIST -o "$INSTALL_DIR/cuda_fshape_scp_dxi_${KGEOM[$i]}${KSIG[$j]}${KSPHERE[$k]}.so" "$LIBKPPATH_DIST/fshape_gpu_dxi.cu";
            #echo "Compiling $LIBKPPATH_DIST/fshape_gpu_df.cu with kernel geom: ${KGEOM[$i]}, kernel sig: ${KSIG[$j]}, kernel sphere: ${KSPHERE[$k]}..."
            #$NVCC -D "__TYPE__=$TYPE" -D "CUDA_BLOCK_SIZE=$BLOCKSIZE" -D "KERNEL_GEOM_TYPE=$i" -D "KERNEL_SIG_TYPE=$j" -D "KERNEL_SPHERE_TYPE=$k" $NVCCFLAGS -I$LIBKPPATH_DIST -o "$INSTALL_DIR/cuda_fshape_scp_df_${KGEOM[$i]}${KSIG[$j]}${KSPHERE[$k]}.so" "$LIBKPPATH_DIST/fshape_gpu_df.cu";
        done;
    done;
done;
