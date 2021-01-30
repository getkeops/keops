
// nvcc -shared -Xcompiler -fPIC -lnvrtc -lcuda test_nvrtc.cu -o test_nvrtc.so

#include <nvrtc.h>
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <ctime>


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


extern "C" __host__ int Eval(bool recompile, const char* ptx_file_name, const char* cu_code, int dimY, int nx, int ny, float *out, float *arg0, float *arg1, float *arg2) {
    
    dim3 blockSize;
    blockSize.x = 32;
	
    dim3 gridSize;
    gridSize.x = nx / blockSize.x + (nx % blockSize.x == 0 ? 0 : 1);
    
    clock_t begin, end;
    
    char *ptx;
    if (recompile) {
        
        begin = clock();
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
        end = clock();
        std::cout << "time for compiling ptx code : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
    
        // write ptx code to file
        begin = clock();
        FILE *ptx_file = fopen(ptx_file_name, "w");
        fputs(ptx, ptx_file);
        fclose(ptx_file);
        end = clock();
        std::cout << "time for writing ptx code to file : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
    }
    else {
        // read ptx code from file
        begin = clock();
        ptx = read_text_file(ptx_file_name);
        end = clock();
        std::cout << "time for reading ptx code from file : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
    }
    
    
    begin = clock();
    // Load the generated PTX and get a handle to the kernel.
    CUdevice cuDevice;
    CUcontext context;
    CUmodule module;
    CUfunction kernel;
    CUDA_SAFE_CALL(cuInit(0));
    CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
    CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
    CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
    CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "GpuConv1DOnDevice"));
    end = clock();
    std::cout << "time for loading the ptx : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
  
        
    void *args[] = { &nx, &ny, &out, &arg0, &arg1, &arg2 };
    begin = clock();
    CUDA_SAFE_CALL(cuLaunchKernel(kernel,
                   gridSize.x, gridSize.y, gridSize.z,    // grid dim
                   blockSize.x, blockSize.y, blockSize.z,   // block dim
                   blockSize.x * dimY * sizeof(float), NULL,             // shared mem and stream
                   args, 0));           // arguments
    CUDA_SAFE_CALL(cuCtxSynchronize());
    end = clock();
    std::cout << "time for executing kernel : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
    
    // Release resources.
    begin = clock();
    CUDA_SAFE_CALL(cuModuleUnload(module));
    CUDA_SAFE_CALL(cuCtxDestroy(context));
    end = clock();
    std::cout << "time for releasing resources : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;                                              

    return 0;
}
