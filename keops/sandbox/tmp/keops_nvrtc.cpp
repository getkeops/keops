
// nvcc -shared -Xcompiler -fPIC -lnvrtc -lcuda keops_nvrtc.cu -o keops_nvrtc.so
// g++ --verbose -L/opt/cuda/lib64 -L/opt/cuda/targets/x86_64-linux/lib/ -I/opt/cuda/targets/x86_64-linux/include/ -shared -fPIC -lcuda -lnvrtc -fpermissive -DMAXIDGPU=0 -DMAXTHREADSPERBLOCK0=1024 -DSHAREDMEMPERBLOCK0=49152  /home/bcharlier/projets/keops/keops/keops/python_engine/compilation/keops_nvrtc.cpp -o /home/bcharlier/projets/keops/keops/keops/python_engine/build/keops_nvrtc.so

#include <nvrtc.h>
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdarg.h>
#include <vector>
#include <ctime>

#define __INDEX__ int
#define C_CONTIGUOUS 1
#define USE_HALF 0

#include "Sizes.h"
#include "Ranges.h"
#include "utils_pe.h"
#include "ranges_utils.h"


#include "CudaSizes.h"
#include <cuda_fp16.h>

template<typename TYPE>
class context {
    public :

int current_device_id = -1;
CUcontext ctx;
CUmodule module;
char *target;



context() {
    
std::cout << "here in context constructor 1" << std::endl;

    target = new char[3+1];
    
    target[0] = 'a';
    target[1] = 'b';
    target[2] = 'c';
    target[3] = '\0';

std::cout << "here in context constructor 2" << std::endl;

    //SetDevice(0);

std::cout << "here in context constructor 3" << std::endl;
}

~context() {

std::cout << "here in context destructor 1" << std::endl;

    
    delete[] target;

std::cout << "here in context destructor 4" << std::endl;

}


};


template class context<float>;

