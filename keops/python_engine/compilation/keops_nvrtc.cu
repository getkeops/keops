
// nvcc -shared -Xcompiler -fPIC -lnvrtc -lcuda keops_nvrtc.cu -o keops_nvrtc.so

#include <nvrtc.h>
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <stdarg.h>

#define NVRTC_SAFE_CALL(x)                                        \
  do {                                                            \
    nvrtcResult result = x;                                       \
    if (result != NVRTC_SUCCESS) {                                \
      std::cerr << "\nerror: " #x " failed with error "           \
                << nvrtcGetErrorString(result) << '\n';           \
      exit(1);                                                    \
    }                                                             \
  } while(0)
#define CUDA_SAFE_CALL(x)                                         \
  do {                                                            \
    CUresult result = x;                                          \
    if (result != CUDA_SUCCESS) {                                 \
      const char *msg;                                            \
      cuGetErrorName(result, &msg);                               \
      std::cerr << "\nerror: " #x " failed with error "           \
                << msg << '\n';                                   \
      exit(1);                                                    \
    }                                                             \
  } while(0)


char* read_text_file(char const* path) {
    char* buffer = 0;
    long length;
    FILE * f = fopen (path, "rb");
    if (f)
    {
      fseek (f, 0, SEEK_END);
      length = ftell (f);
      fseek (f, 0, SEEK_SET);
      buffer = (char*)malloc ((length+1)*sizeof(char));
      if (buffer)
      {
        fread (buffer, sizeof(char), length, f);
      }
      fclose (f);
    }
    buffer[length] = '\0';
    return buffer;
}


extern "C" __host__ int Compile(const char* ptx_file_name, const char* cu_code) {

    char *ptx;

    nvrtcProgram prog;

    NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog,         // prog
                   cu_code,         // buffer
                   NULL,    // name
                   0,             // numHeaders
                   NULL,          // headers
                   NULL));        // includeNames

    const char *opts[] = {};
    nvrtcResult compileResult = nvrtcCompileProgram(prog,  // prog
                                              0,     // numOptions
                                              opts); // options
              
    // Obtain compilation log from the program.
    size_t logSize;
    NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
    char *log = new char[logSize];
    NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));
    std::cout << log << '\n';
    delete[] log;
    if (compileResult != NVRTC_SUCCESS) {
        exit(1);
    }

    // Obtain PTX from the program.
    size_t ptxSize;
    NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
    ptx = new char[ptxSize];
    NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));
    // Destroy the program.
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
    
    // write ptx code to file
    FILE *ptx_file = fopen(ptx_file_name, "w");
    fputs(ptx, ptx_file);
    fclose(ptx_file);

    return 0;
}

template < typename TYPE >
int get_size(TYPE *shape) {
    int ndims = shape[0];
    int size = 1;
    for (int k=0; k<ndims; k++)
        size *= shape[k+1];
    return size;
}


template < typename TYPE >
__host__ void load_args_FromDevice(void*& p_data, TYPE* out, TYPE*& out_d, int nargs, TYPE** arg, TYPE**& arg_d) {
    cudaMalloc(&p_data, sizeof(TYPE*) * nargs);
    out_d = out;
    arg_d = (TYPE **) p_data;
    // copy array of pointers
    cudaMemcpy(arg_d, arg, nargs * sizeof(TYPE *), cudaMemcpyHostToDevice);
}


template < typename TYPE >
__host__ void load_args_FromHost(void*& p_data, TYPE* out, TYPE*& out_d, int nargs, TYPE** arg, TYPE**& arg_d, int**& argshape, int sizeout) {
    int sizes[nargs];
    int totsize = sizeout;
    for (int k=0; k<nargs; k++) {
        sizes[k] = get_size(argshape[k]);
        totsize += sizes[k];
    }
    cudaMalloc(&p_data, sizeof(TYPE *) * nargs + sizeof(TYPE) * totsize);
    
    arg_d = (TYPE**) p_data;
    TYPE *dataloc = (TYPE *) (arg_d + nargs);
    
    // host array of pointers to device data
    TYPE *ph[nargs];
                
    out_d = dataloc;
    dataloc += sizeout;
    for (int k=0; k<nargs; k++) {
        ph[k] = dataloc;
        cudaMemcpy(dataloc, arg[k], sizeof(TYPE) * sizes[k], cudaMemcpyHostToDevice);
        dataloc += sizes[k];
    }
    
    // copy array of pointers
    cudaMemcpy(arg_d, ph, nargs * sizeof(TYPE *), cudaMemcpyHostToDevice);
}

template < typename TYPE >
__host__ int launch_keops(const char* ptx_file_name, int tagHostDevice, int dimY, int nx, int ny, int device_id, int tagI, 
                                        int **ranges, int *shapeout, TYPE *out, int nargs, TYPE **arg, int **argshape) {
    
    CUdevice cuDevice;
    CUDA_SAFE_CALL(cuInit(0));
    CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, device_id));
    
    cudaSetDevice(device_id);
    
    if (tagI==1) {
        int tmp = ny;
        ny = nx;
        nx = tmp;
    }
    
    void *p_data;
    TYPE *out_d;
    TYPE **arg_d;
    int sizeout = get_size(shapeout);
    
    if(tagHostDevice==1)
        load_args_FromDevice(p_data, out, out_d, nargs, arg, arg_d);
    else
        load_args_FromHost(p_data, out, out_d, nargs, arg, arg_d, argshape, sizeout);
    
    dim3 blockSize;
    blockSize.x = 192;
	
    dim3 gridSize;
    gridSize.x = nx / blockSize.x + (nx % blockSize.x == 0 ? 0 : 1);

    
    char *ptx;
    ptx = read_text_file(ptx_file_name);
    
    CUmodule module;
    CUfunction kernel;
    
    CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
    CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "GpuConv1DOnDevice"));

    void *kernel_params[4];
    kernel_params[0] = &nx;
    kernel_params[1] = &ny;
    kernel_params[2] = &out_d;
    kernel_params[3] = &arg_d;

    CUDA_SAFE_CALL(cuLaunchKernel(kernel,
                   gridSize.x, gridSize.y, gridSize.z,    // grid dim
                   blockSize.x, blockSize.y, blockSize.z,   // block dim
                   blockSize.x * dimY * sizeof(TYPE), NULL,             // shared mem and stream
                   kernel_params, 0));           // arguments

    CUDA_SAFE_CALL(cuCtxSynchronize());

    CUDA_SAFE_CALL(cuModuleUnload(module));
    
    // Send data from device to host.

    if(tagHostDevice==0)
        cudaMemcpy(out, out_d, sizeof(TYPE) * sizeout, cudaMemcpyDeviceToHost);

    cudaFree(p_data);

    return 0;
}



extern "C" __host__ int launch_keops_float(const char* ptx_file_name, int tagHostDevice, int dimY, int nx, int ny, int device_id, int tagI, 
                                        int **ranges, int *shapeout, float *out, int nargs, ...) {
    // reading arguments
    va_list ap;
    va_start(ap, nargs);
    float *arg[nargs];
    for (int i=0; i<nargs; i++)
        arg[i] = va_arg(ap, float*);
    int *argshape[nargs];
    for (int i=0; i<nargs; i++)
        argshape[i] = va_arg(ap, int*);
    va_end(ap);
    
    return launch_keops(ptx_file_name, tagHostDevice, dimY, nx, ny, device_id, tagI, 
                                        ranges, shapeout, out, nargs, arg, argshape);
                                                                        
}

extern "C" __host__ int launch_keops_double(const char* ptx_file_name, int tagHostDevice, int dimY, int nx, int ny, int device_id, int tagI, 
                                        int **ranges, int *shapeout, double *out, int nargs, ...) {
    // reading arguments
    va_list ap;
    va_start(ap, nargs);
    double *arg[nargs];
    for (int i=0; i<nargs; i++)
        arg[i] = va_arg(ap, double*);
    int *argshape[nargs];
    for (int i=0; i<nargs; i++)
        argshape[i] = va_arg(ap, int*);
    va_end(ap);
    
    return launch_keops(ptx_file_name, tagHostDevice, dimY, nx, ny, device_id, tagI, 
                                        ranges, shapeout, out, nargs, arg, argshape);
                                                                        
}
