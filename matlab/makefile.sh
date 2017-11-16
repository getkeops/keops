#!/bin/bash

# --------- # PATH: --------- #

MATLABROOT="/usr/local/MATLAB/R2014a"
CUDAROOT="/usr/lib/x86_64-linux-gnu/"
MEXC="$MATLABROOT/bin/mex"
CC="/usr/bin/gcc"
NVCC="/usr/bin/nvcc"
LIBKPROOT=".."

SRC_DIR="$LIBKPROOT/matlab/matkp/src"
INSTALL_DIR="$LIBKPROOT/matlab/matkp/mex"

# --------- # GPU PARAMETERS: --------- #

COMPUTECAPABILITY=35
USE_DOUBLE=0
BLOCKSIZE=192

# --------- # NVCC PARAMETERS: --------- #

NVCCFLAGS="-ccbin=$CC -arch=sm_$COMPUTECAPABILITY -Xcompiler -fPIC"
MEXPATH="-I$MATLABROOT/extern/include"
LIBKPPATH="-I$LIBKPROOT/cuda/convolutions -I$LIBKPROOT/cuda/shape_distance"

# --------- C COMPILATION PARAMETERS: --------- #
COPTIMFLAG="-O3" 
CLIB="-L$CUDAROOT -lcudart"

# --------- FSHAPES DISTANCES PARAMETERS: --------- #
KGEOM=( "gaussian" "cauchy" )
KSIG=( "gaussian" "cauchy" )
KSPHERE=("gaussian_unoriented" "binet" "gaussian_oriented" "linear")

KERNEL_GEOM=(0 1)
KERNEL_SIG=(0 1)
KERNEL_SPHERE=(0 1 2 3)



#clean
rm -f *.o;
rm -f $INSTALL_DIR/*.mexa64;

#create object file with nvcc
echo "Compiling $SRC_DIR/cudaconv.cu..."
$NVCC -c -D "USE_DOUBLE_PRECISION=$USE_DOUBLE" -D "CUDA_BLOCK_SIZE=$BLOCKSIZE" $SRC_DIR/cudaconv.cu $NVCCFLAGS $MEXPATH $LIBKPPATH -o cudaconv.o;
echo "Compiling $SRC_DIR/cudagrad1conv.cu..."
$NVCC -c -D "USE_DOUBLE_PRECISION=$USE_DOUBLE" -D "CUDA_BLOCK_SIZE=$BLOCKSIZE" $SRC_DIR/cudagrad1conv.cu $NVCCFLAGS $MEXPATH $LIBKPPATH -o cudagrad1conv.o;


for i in ${KERNEL_GEOM[@]}; do 
    for j in ${KERNEL_SIG[@]}; do
        for k in ${KERNEL_SPHERE[@]}; do
            echo "Compiling $SRC_DIR/cudafshape.cu with kernel geom: ${KGEOM[$i]}, kernel sig: ${KSIG[$j]}, kernel sphere: ${KSPHERE[$k]}..."
            $NVCC -c -D "USE_DOUBLE_PRECISION=$USE_DOUBLE" -D "CUDA_BLOCK_SIZE=$BLOCKSIZE" -D "KERNEL_GEOM_TYPE=$i" -D "KERNEL_SIG_TYPE=$j" -D "KERNEL_SPHERE=$k" $SRC_DIR/cudafshape.cu $NVCCFLAGS $MEXPATH $LIBKPPATH -o ./cuda_fshape_scp_${KGEOM[$i]}${KSIG[$j]}${KSPHERE[$k]}.o;
            #echo "Compiling $SRC_DIR/cudafshape_dx.cu with kernel geom: ${KGEOM[$i]}, kernel sig: ${KSIG[$j]}, kernel sphere: ${KSPHERE[$k]}..."
            #$NVCC -c -D "USE_DOUBLE_PRECISION=$USE_DOUBLE" -D "CUDA_BLOCK_SIZE=$BLOCKSIZE" -D "KERNEL_GEOM_TYPE=$i" -D "KERNEL_SIG_TYPE=$j" -D "KERNEL_SPHERE=$k" $SRC_DIR/cudafshape_dx.cu $NVCCFLAGS $MEXPATH $LIBKPPATH -o ./cudafshape_scp_${KGEOM[$i]}${KSIG[$j]}${KSPHERE[$k]}_dx.o;
            #echo "Compiling $SRC_DIR/cudafshape_dxi.cu with kernel geom: ${KGEOM[$i]}, kernel sig: ${KSIG[$j]}, kernel sphere: ${KSPHERE[$k]}..."
            #$NVCC -c -D "USE_DOUBLE_PRECISION=$USE_DOUBLE" -D "CUDA_BLOCK_SIZE=$BLOCKSIZE" -D "KERNEL_GEOM_TYPE=$i" -D "KERNEL_SIG_TYPE=$j" -D "KERNEL_SPHERE=$k" $SRC_DIR/cudafshape_dxi.cu $NVCCFLAGS $MEXPATH $LIBKPPATH -o ./cudafshape_scp_${KGEOM[$i]}${KSIG[$j]}${KSPHERE[$k]}_dxi.o;
            #echo "Compiling $SRC_DIR/cudafshape_df.cu with kernel geom: ${KGEOM[$i]}, kernel sig: ${KSIG[$j]}, kernel sphere: ${KSPHERE[$k]}..."
            #$NVCC -c -D "USE_DOUBLE_PRECISION=$USE_DOUBLE" -D "CUDA_BLOCK_SIZE=$BLOCKSIZE" -D "KERNEL_GEOM_TYPE=$i" -D "KERNEL_SIG_TYPE=$j" -D "KERNEL_SPHERE=$k" $SRC_DIR/cudafshape_df.cu $NVCCFLAGS $MEXPATH $LIBKPPATH -o ./cudafshape_scp_${KGEOM[$i]}${KSIG[$j]}${KSPHERE[$k]}_df.o;
        done;
    done;
done;



#mex complilation
for i in `ls *.o`;do $MEXC GCC=$CC COPTIMFLAGS=$COPTIMFLAG $i $CLIB;done

#clean
rm -f *.o;

# install   
mkdir -p "$INSTALL_DIR"

for i in `ls *.mexa64`;do 
    mv $i "$INSTALL_DIR";
    echo "$i successfully installed"
done

