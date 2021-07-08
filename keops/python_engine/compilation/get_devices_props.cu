#include <cstdio>

int main() {
    int count = 0;
    if (cudaSuccess != cudaGetDeviceCount(&count)) 
        return -1;
    if (count == 0) 
        return -1;
    std::printf("-DMAXIDGPU=%d;",count-1);
    for (int device = 0; device < count; ++device) {
        cudaDeviceProp prop;
        if (cudaSuccess == cudaGetDeviceProperties(&prop, device))
            std::printf("-DMAXTHREADSPERBLOCK%d=%d;-DSHAREDMEMPERBLOCK%d=%d;", 
                        device, 
                        (int)prop.maxThreadsPerBlock, 
                        device, (int)prop.sharedMemPerBlock);
    }
    return 0;
}
