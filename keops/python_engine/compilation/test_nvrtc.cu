
// nvcc -shared -Xcompiler -fPIC -lnvrtc -lcuda test_nvrtc.cu -o test_nvrtc.so

#include <nvrtc.h>
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <stdarg.h>

#define TIMEIT 1
#if TIMEIT
#include <ctime>
#endif


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
        
#if TIMEIT
    clock_t begin, end;
    
    begin = clock();
#endif

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
    
#if TIMEIT
    end = clock();
    std::cout << "time for compiling ptx code : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
    
    begin = clock();
#endif
    // write ptx code to file
    FILE *ptx_file = fopen(ptx_file_name, "w");
    fputs(ptx, ptx_file);
    fclose(ptx_file);
#if TIMEIT
    end = clock();
    std::cout << "time for writing ptx code to file : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
#endif

    return 0;
}


extern "C" __host__ int Eval(const char* ptx_file_name, int dimY, int nx, int ny, float *out, int nargs, ...) {
    
    float *arg[nargs];
    va_list ap;
    va_start(ap, nargs);
    for (int i=0; i<nargs; i++)
        arg[i] = va_arg(ap, float*);
    va_end(ap);
    
    dim3 blockSize;
    blockSize.x = 32;
	
    dim3 gridSize;
    gridSize.x = nx / blockSize.x + (nx % blockSize.x == 0 ? 0 : 1);

#if TIMEIT    
    clock_t begin, end;
    begin = clock();
#endif
    
    char *ptx;
    
    // read ptx code from file
    ptx = read_text_file(ptx_file_name);
#if TIMEIT    
    end = clock();
    std::cout << "time for reading ptx code from file : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    begin = clock();
#endif
    // Load the generated PTX and get a handle to the kernel.
    CUdevice cuDevice;
    //CUcontext context;
    CUmodule module;
    CUfunction kernel;

    CUDA_SAFE_CALL(cuInit(0));

    CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));

    //CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
#if TIMEIT  
    end = clock();
    std::cout << "time for loading the ptx (part 1) : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
    
    begin = clock();
#endif
    CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
#if TIMEIT 
    end = clock();
    std::cout << "time for loading the ptx (part 2) : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
    
    begin = clock();
#endif
    CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "GpuConv1DOnDevice"));
#if TIMEIT
    end = clock();
    std::cout << "time for loading the ptx (part 3) : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
  
    begin = clock();
#endif
    void *kernel_params[nargs+3];
    kernel_params[0] = &nx;
    kernel_params[1] = &ny;
    kernel_params[2] = &out;
    for (int i=0; i<nargs; i++)
        kernel_params[i+3] = &arg[i];
    CUDA_SAFE_CALL(cuLaunchKernel(kernel,
                   gridSize.x, gridSize.y, gridSize.z,    // grid dim
                   blockSize.x, blockSize.y, blockSize.z,   // block dim
                   blockSize.x * dimY * sizeof(float), NULL,             // shared mem and stream
                   kernel_params, 0));           // arguments
    CUDA_SAFE_CALL(cuCtxSynchronize());
#if TIMEIT
    end = clock();
    std::cout << "time for executing kernel : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
    
    begin = clock();
#endif
    // Release resources.
    CUDA_SAFE_CALL(cuModuleUnload(module));
    //CUDA_SAFE_CALL(cuCtxDestroy(context));
#if TIMEIT
    end = clock();
    std::cout << "time for releasing resources : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;                                              
#endif
    return 0;
}
