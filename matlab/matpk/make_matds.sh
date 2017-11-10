# --------- # PATH: --------- #

MATLABROOT="/usr/local/MATLAB/R2014a"
CUDAROOT="/usr/lib/x86_64-linux-gnu/"
MEXC="$MATLABROOT/bin/mex"
CC="/usr/bin/gcc"
NVCC="/usr/bin/nvcc"
LIBDSROOT="../../"

INSTALL_DIR="$LIBDSROOT/matlab/matpk/mex"

# --------- # GPU PARAMETERS: --------- #

COMPUTECAPABILITY=35
USE_DOUBLE=0
BLOCKSIZE=192

# --------- # NVCC PARAMETERS: --------- #

NVCCFLAGS="-ccbin=$CC -arch=sm_$COMPUTECAPABILITY -Xcompiler -fPIC"
MEXPATH="-I$MATLABROOT/extern/include"
LIBDSPATH="-I$LIBDSROOT/cuda"

# --------- C COMPILATION PARAMETERS: --------- #
COPTIMFLAG="-O3" 
CLIB="-L$CUDAROOT -lcudart"


#clean
rm -f *.o;
rm -f $INSTALL_DIR/*.mexa64;

#create object file with nvcc
$NVCC -c -D "USE_DOUBLE_PRECISION=$USE_DOUBLE" -D "CUDA_BLOCK_SIZE=$BLOCKSIZE" ./src/cudaconv.cu $NVCCFLAGS $MEXPATH $LIBDSPATH -o cudaconv.o;
$NVCC -c -D "USE_DOUBLE_PRECISION=$USE_DOUBLE" -D "CUDA_BLOCK_SIZE=$BLOCKSIZE" ./src/cudagrad1conv.cu $NVCCFLAGS $MEXPATH $LIBDSPATH -o cudagrad1conv.o;


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

