#include <nvrtc.h>
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <ctime>

#define NUM_THREADS 128
#define NUM_BLOCKS 32


char* read_text_file(char const* path) {
    char* buffer = 0;
    long length;
    FILE * f = fopen (path, "rb"); //was "rb"
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


extern "C" __host__ int Eval(const char* cu_code, int dimY, int nx, int ny, float *out, float *arg0, float *arg1, float *arg2) {
    
    std::cout << cu_code << "ZZZZZ" << std::endl;
    
    dim3 blockSize;
    blockSize.x = 32;
	
    dim3 gridSize;
    gridSize.x = nx / blockSize.x + (nx % blockSize.x == 0 ? 0 : 1);
    
    clock_t begin, end;
    
    
    char ptx_file_name[20] = "test.ptx";
    
    bool recompile = true;
    char *ptx;
    if (recompile) {
        
        begin = clock();
        nvrtcProgram prog;
    
        nvrtcCreateProgram(&prog,         // prog
                       cu_code,         // buffer
                       NULL,    // name
                       0,             // numHeaders
                       NULL,          // headers
                       NULL);        // includeNames
    
        const char *opts[] = {"-O3"};
        nvrtcResult compileResult = nvrtcCompileProgram(prog,  // prog
                                                  1,     // numOptions
                                                  opts); // options
                                                  
        size_t ptxSize;
        nvrtcGetPTXSize(prog, &ptxSize);
        ptx = new char[ptxSize];
        nvrtcGetPTX(prog, ptx);
        // Destroy the program.
        nvrtcDestroyProgram(&prog);
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
    // Load the generated PTX and get a handle to the SAXPY kernel.
    CUdevice cuDevice;
    CUcontext context;
    CUmodule module;
    CUfunction kernel;
    cuInit(0);
    cuDeviceGet(&cuDevice, 0);
    cuCtxCreate(&context, 0, cuDevice);
    cuModuleLoadDataEx(&module, ptx, 0, 0, 0);
    cuModuleGetFunction(&kernel, module, "GpuConv1DOnDevice");
    end = clock();
    std::cout << "time for loading the ptx : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
  
        
    void *args[] = { &nx, &ny, &out, &arg0, &arg1, &arg1, &arg2 };
    begin = clock();
    cuLaunchKernel(kernel,
                   gridSize.x, gridSize.y, gridSize.z,    // grid dim
                   blockSize.x, blockSize.y, blockSize.z,   // block dim
                   blockSize.x * dimY * sizeof(float), NULL,             // shared mem and stream
                   args, 0);           // arguments
    cuCtxSynchronize();
    end = clock();
    std::cout << "time for executing kernel : " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
    
    // Release resources.
    cuModuleUnload(module);
    cuCtxDestroy(context);
                                                  

    return 0;
}
