# --------- # PATH: --------- #

MATLABROOT="/usr/local/MATLAB/R2014a"
CUDAROOT="/usr/lib/x86_64-linux-gnu/"
MEXC="$MATLABROOT/bin/mex"
CC="/usr/bin/gcc"
NVCC="/usr/bin/nvcc"
LIBKPROOT="../../"

INSTALL_DIR="$LIBKPROOT/matlab/matpk/mex"

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
KVAR=("gaussian_unoriented" "binet" "gaussian_oriented" "linear")

KERNEL_GEOM=(0)
KERNEL_SIG=(0 1)
KERNEL_VAR=(0 2)



#clean
rm -f *.o;
rm -f $INSTALL_DIR/*.mexa64;

#create object file with nvcc
#$NVCC -c -D "USE_DOUBLE_PRECISION=$USE_DOUBLE" -D "CUDA_BLOCK_SIZE=$BLOCKSIZE" ./src/cudaconv.cu $NVCCFLAGS $MEXPATH $LIBKPPATH -o cudaconv.o;
#$NVCC -c -D "USE_DOUBLE_PRECISION=$USE_DOUBLE" -D "CUDA_BLOCK_SIZE=$BLOCKSIZE" ./src/cudagrad1conv.cu $NVCCFLAGS $MEXPATH $LIBKPPATH -o cudagrad1conv.o;


for i in ${KERNEL_GEOM[@]}; do 
    for j in ${KERNEL_SIG[@]}; do
        for k in ${KERNEL_VAR[@]}; do
            $NVCC -c -D "USE_DOUBLE_PRECISION=$USE_DOUBLE" -D "CUDA_BLOCK_SIZE=$BLOCKSIZE" -D "KERNEL_GEOM_TYPE=$i" -D "KERNEL_SIG_TYPE=$j" -D "KERNEL_VAR_TYPE=$k" ./src/cudafshape.cu $NVCCFLAGS $MEXPATH $LIBKPPATH -o ./cudafshape_scp_${KGEOM[$i]}${KSIG[$j]}${KVAR[$k]}.o;
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

