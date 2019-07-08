
// small program to print some Gpu properties
// compile with nvcc GetGpuProps.cu -o build/GetGpuProps

#include <cstdio>
int main()
{
  int count = 0;
  if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;
  if (count == 0) return -1;
  std::printf("\nNumber of Cuda Gpu devices: %d\n\n",count-1);
  cudaDeviceProp prop;
  for (int device = 0; device < count; ++device)
  {
    std::printf("Device %d:\n",device);
    if (cudaSuccess == cudaGetDeviceProperties(&prop, device))
      std::printf("Max threads per block: %d\nShared memory per block: %d\n\n", (int)prop.maxThreadsPerBlock, (int)prop.sharedMemPerBlock);
  }
  return 0;
}


