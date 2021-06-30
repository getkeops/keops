
// nvcc -shared -Xcompiler -fPIC -lnvrtc -lcuda keops_nvrtc.cu -o keops_nvrtc.so

#include <nvrtc.h>
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <stdarg.h>

#define __INDEX__ int //int32_t

#define C_CONTIGUOUS 1
#define USE_HALF 0

#include "Sizes.h"
#include "Ranges.h"

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


int broadcast_index(int i, int nbatchdims, int *full_shape, int *shape) {
    int M_N = shape[nbatchdims];
    int res = i % M_N, step = M_N, full_step = M_N;
    for (int b = nbatchdims; b > 0; b--) {
        if (shape[b - 1] != 1) {
            res += ((i / full_step) % shape[b - 1]) * step;
        }
        full_step *= full_shape[b - 1];
        step *= shape[b - 1];
    }
    return res;
}
                
void vect_broadcast_index(int i, int nbatchdims, int nvars, int *full_shape,
                          int *reduced_shapes, int *out, int add_offset = 0) {
    for (int k = 0; k < nvars; k++) {
        out[k] = add_offset + broadcast_index(i, nbatchdims, full_shape, reduced_shapes + (nbatchdims + 1) * k);
    }
}

void fill_shapes(int nbatchdims, int *shapes, int *shapes_i, int *shapes_j, int *shapes_p,
                 int tagJ, int sizei, int sizej, int sizep, int *indsi, int *indsj, int *indsp) {

  sizei += 1;

  const int tagIJ = tagJ; // 1 if the reduction is made "over j", 0 if it is made "over i"

  // Separate and store the shapes of the "i" and "j" variables + parameters --------------
  //
  // N.B.: If tagIJ == 1, the reduction is made over 'j', which is the default mode.
  //       However, if tagIJ == 0, the reduction is performed over the 'i' variables:
  //       since "shape" does not change, we must adapt the adress at which we pick information...
  //
  // shapes is an array of size (1+nargs)*(nbatchdims+3), which looks like:
  // [ A, .., B, M, N, D_out]  -> output
  // [ A, .., B, M, 1, D_1  ]  -> "i" variable
  // [ A, .., B, 1, N, D_2  ]  -> "j" variable
  // [ A, .., B, 1, 1, D_3  ]  -> "parameter"
  // [ A, .., 1, M, 1, D_4  ]  -> N.B.: we support broadcasting on the batch dimensions!
  // [ 1, .., 1, M, 1, D_5  ]  ->      (we'll just ask users to fill in the shapes with *explicit* ones)

  // First, we fill shapes_i with the "relevant" shapes of the "i" variables,
  // making it look like, say:
  // [ A, .., B, M]
  // [ A, .., 1, M]
  // [ A, .., A, M]
  for (int k = 0; k < (sizei - 1); k++) {  // k-th line
    for (int l = 0; l < nbatchdims; l++) {  // l-th column
      shapes_i[k * (nbatchdims + 1) + l] = shapes[(1 + indsi[k]) * (nbatchdims + 3) + l];
    }
    shapes_i[k * (nbatchdims + 1) + nbatchdims] =
        shapes[(1 + indsi[k]) * (nbatchdims + 3) + nbatchdims + 1 - tagIJ];
  }

  // Then, we do the same for shapes_j, but with "N" instead of "M":
  for (int k = 0; k < sizej; k++) {  // k-th line
    for (int l = 0; l < nbatchdims; l++) {  // l-th column
      shapes_j[k * (nbatchdims + 1) + l] = shapes[(1 + indsj[k]) * (nbatchdims + 3) + l];
    }
    shapes_j[k * (nbatchdims + 1) + nbatchdims] = shapes[(1 + indsj[k]) * (nbatchdims + 3) + nbatchdims + tagIJ];
  }

  // And finally for the parameters, with "1" instead of "M":
  for (int k = 0; k < sizep; k++) {  // k-th line
    for (int l = 0; l < nbatchdims; l++) {  // l-th column
      shapes_p[k * (nbatchdims + 1) + l] = shapes[(1 + indsp[k]) * (nbatchdims + 3) + l];
    }
    shapes_p[k * (nbatchdims + 1) + nbatchdims] = 1;
  }

}




int* build_offset_tables( int nbatchdims, int *shapes, int nblocks, __INDEX__ *lookup_h,
                          int sizei, int sizej, int sizep, int *indsi, int *indsj, int *indsp,
                          int tagJ) {

        // Support for broadcasting over batch dimensions =============================================
    
        int sizevars = sizei + sizej + sizep;
    
        // Separate and store the shapes of the "i" and "j" variables + parameters --------------
        //
        // shapes is an array of size (1+nargs)*(nbatchdims+3), which looks like:
        // [ A, .., B, M, N, D_out]  -> output
        // [ A, .., B, M, 1, D_1  ]  -> "i" variable
        // [ A, .., B, 1, N, D_2  ]  -> "j" variable
        // [ A, .., B, 1, 1, D_3  ]  -> "parameter"
        // [ A, .., 1, M, 1, D_4  ]  -> N.B.: we support broadcasting on the batch dimensions!
        // [ 1, .., 1, M, 1, D_5  ]  ->      (we'll just ask users to fill in the shapes with *explicit* ones)
    
        int shapes_i[sizei*(nbatchdims+1)], shapes_j[sizej*(nbatchdims+1)], shapes_p[sizep*(nbatchdims+1)];
    
        // First, we fill shapes_i with the "relevant" shapes of the "i" variables,
        // making it look like, say:
        // [ A, .., B, M]
        // [ A, .., 1, M]
        // [ A, .., A, M]
        // Then, we do the same for shapes_j, but with "N" instead of "M".
        // And finally for the parameters, with "1" instead of "M".
        fill_shapes(nbatchdims, shapes, shapes_i, shapes_j, shapes_p, tagJ, sizei, sizej, sizep, indsi, indsj, indsp);
    
        int tagIJ = tagJ; // 1 if the reduction is made "over j", 0 if it is made "over i"
        int M = shapes[nbatchdims], N = shapes[nbatchdims+1];

        // We create a lookup table, "offsets", of shape (nblocks, SIZEVARS) --------
        int *offsets_h = NULL, *offsets_d = NULL;
    
        offsets_h = new int[nblocks * sizevars] ;

        for (int k=0; k < nblocks; k++) {
            int range_id = (int) lookup_h[3*k] ;
            int start_x  = tagIJ ? range_id * M : range_id * N;
            int start_y  = tagIJ ? range_id * N : range_id * M;

            int patch_offset = (int) (lookup_h[3*k+1]-start_x);
            
            vect_broadcast_index(start_x, nbatchdims, sizei, shapes, shapes_i, offsets_h + k*sizevars, patch_offset);
            vect_broadcast_index(start_y, nbatchdims, sizej,   shapes, shapes_j, offsets_h + k*sizevars + sizei);
            vect_broadcast_index(range_id, nbatchdims, sizep, shapes, shapes_p, offsets_h + k*sizevars + sizei + sizej);
        }

        cudaMalloc((int**)&offsets_d, sizeof(int)*nblocks*sizevars);
        cudaMemcpy(offsets_d, offsets_h, sizeof(int)*nblocks*sizevars, cudaMemcpyHostToDevice);
    
        delete [] offsets_h;
        return offsets_d;
}


void range_preprocess(int& nblocks, int tagI, int nranges_x, int nranges_y, __INDEX__ **castedranges,
                      int nbatchdims, __INDEX__*& slices_x_d, __INDEX__*& ranges_y_d,
                      __INDEX__*& lookup_d, int*& offsets_d, int blockSize_x,
                      int *indsi, int *indsj, int *indsp, int *shapes) {

        // Ranges pre-processing... ==================================================================
        
        // N.B.: In the following code, we assume that the x-ranges do not overlap.
        //       Otherwise, we'd have to assume that DIMRED == DIMOUT
        //       or allocate a buffer of size nx * DIMRED. This may be done in the future.
        // Cf. reduction.h: 
        //    FUN::tagJ = 1 for a reduction over j, result indexed by i
        //    FUN::tagJ = 0 for a reduction over i, result indexed by j
        
        int tagJ = 1-tagI;
        int nranges = tagJ ? nranges_x : nranges_y ;

        __INDEX__ *ranges_x = tagJ ? castedranges[0] : castedranges[3] ;
        __INDEX__ *slices_x = tagJ ? castedranges[1] : castedranges[4] ;
        __INDEX__ *ranges_y = tagJ ? castedranges[2] : castedranges[5] ;
    
        __INDEX__ *ranges_x_h = NULL;
    
        // The code below needs a pointer to ranges_x on *host* memory,  -------------------
        // as well as pointers to slices_x and ranges_y on *device* memory.
        // -> Depending on the "ranges" location, we'll copy ranges_x *or* slices_x and ranges_y
        //    to the appropriate memory:
        bool ranges_on_device = (nbatchdims==0);  
        // N.B.: We only support Host ranges with Device data when these ranges were created 
        //       to emulate block-sparse reductions.
    
        if ( ranges_on_device ) {  // The ranges are on the device
            ranges_x_h = new __INDEX__[2*nranges] ;
            // Send data from device to host.
            cudaMemcpy(ranges_x_h, ranges_x, sizeof(__INDEX__)*2*nranges, cudaMemcpyDeviceToHost);
            slices_x_d = slices_x;
            ranges_y_d = ranges_y;
        } else {  // The ranges are on host memory; this is typically what happens with **batch processing**,
                // with ranges generated by keops_io.h:
            ranges_x_h = ranges_x;
    
            // Copy "slices_x" to the device:
           
            cudaMalloc((__INDEX__**)&slices_x_d, sizeof(__INDEX__)*nranges);
            cudaMemcpy(slices_x_d, slices_x, sizeof(__INDEX__)*nranges, cudaMemcpyHostToDevice);
    
            // Copy "redranges_y" to the device: with batch processing, we KNOW that they have the same shape as ranges_x
            cudaMalloc((__INDEX__**)&ranges_y_d, sizeof(__INDEX__)*2*nranges);
            cudaMemcpy(ranges_y_d, ranges_y, sizeof(__INDEX__)*2*nranges, cudaMemcpyHostToDevice);
        }
    
        // Computes the number of blocks needed ---------------------------------------------
        nblocks = 0;
        int len_range = 0;
        for(int i=0; i<nranges ; i++){
            len_range = ranges_x_h[2*i+1] - ranges_x_h[2*i] ;
            nblocks += (len_range/blockSize_x) + (len_range%blockSize_x==0 ? 0 : 1) ;
        }
    
        // Create a lookup table for the blocks --------------------------------------------
        __INDEX__ *lookup_h = NULL;
        lookup_h = new __INDEX__[3*nblocks] ;
        int index = 0;
        for(int i=0; i<nranges ; i++){
            len_range = ranges_x_h[2*i+1] - ranges_x_h[2*i] ;
            for(int j=0; j<len_range ; j+=blockSize_x) {
                lookup_h[3*index]   = i;
                lookup_h[3*index+1] = ranges_x_h[2*i] + j;
                lookup_h[3*index+2] = ranges_x_h[2*i] + j + ::std::min((int) blockSize_x, len_range-j ) ;
               index++;
            }
        }
    
        // Load the table on the device -----------------------------------------------------
        cudaMalloc((__INDEX__**)&lookup_d, sizeof(__INDEX__)*3*nblocks);
        cudaMemcpy(lookup_d, lookup_h, sizeof(__INDEX__)*3*nblocks, cudaMemcpyHostToDevice);
    
        // Support for broadcasting over batch dimensions =============================================
    
        // We create a lookup table, "offsets", of shape (nblock, SIZEVARS):
        
        int sizei = indsi[0];
        int sizej = indsj[0];
        int sizep = indsp[0];
    
        if (nbatchdims > 0) {
            offsets_d = build_offset_tables( nbatchdims, shapes, nblocks, lookup_h, 
                                             sizei, sizej, sizep, indsi, indsj, indsp, tagJ);
        }
}


template < typename TYPE >
__host__ int launch_keops(const char* ptx_file_name, int tagHostDevice, int dimY, int nx, int ny, int device_id, int tagI, 
                                        int *indsi, int *indsj, int *indsp,
                                        int dimout, 
                                        int *dimsx, int *dimsy, int *dimsp,
                                        int **ranges, int *shapeout, TYPE *out, int nargs, TYPE **arg, int **argshape) {
    
    CUdevice cuDevice;
    CUDA_SAFE_CALL(cuInit(0));
    CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, device_id));
    
    cudaSetDevice(device_id);
    
    Sizes<TYPE> SS(nargs, arg, argshape, nx, ny, tagI,
                   dimout, 
                   indsi, indsj, indsp,
                   dimsx, dimsy, dimsp);
                        
    #if USE_HALF
        SS.switch_to_half2_indexing();
    #endif

    Ranges<TYPE> RR(SS, ranges);
    
    nx = SS.nx;
    ny = SS.ny;  
    
    dim3 blockSize;
    blockSize.x = 32;
	
    dim3 gridSize;
    
    int nblocks;
    
    if (tagI==1) {
        int tmp = ny;
        ny = nx;
        nx = tmp;
    }

    
    __INDEX__ *lookup_d = NULL, *slices_x_d = NULL, *ranges_y_d = NULL;
    int *offsets_d = NULL;
    
    if (RR.tagRanges==1) {
        range_preprocess(nblocks, tagI, RR.nranges_x, RR.nranges_y, RR.castedranges,
                         SS.nbatchdims, slices_x_d, ranges_y_d, lookup_d,
                         offsets_d,
                         blockSize.x, indsi, indsj, indsp, SS.shapes);
    }
    
    void *p_data;
    TYPE *out_d;
    TYPE **arg_d;
    int sizeout = get_size(shapeout);
    
    if(tagHostDevice==1)
        load_args_FromDevice(p_data, out, out_d, nargs, arg, arg_d);
    else
        load_args_FromHost(p_data, out, out_d, nargs, arg, arg_d, argshape, sizeout);
    
    
    char *ptx;
    ptx = read_text_file(ptx_file_name);
    
    CUmodule module;
    CUfunction kernel;
    
    CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
    
    if (RR.tagRanges==1) {
        // ranges mode
        
        gridSize.x = nblocks;
        
        CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "GpuConv1DOnDevice_ranges_NoChunks"));

        void *kernel_params[9];
        kernel_params[0] = &nx;
        kernel_params[1] = &ny;
        kernel_params[2] = &SS.nbatchdims;
        kernel_params[3] = &offsets_d;
        kernel_params[4] = &lookup_d;
        kernel_params[5] = &slices_x_d;
        kernel_params[6] = &ranges_y_d;
        kernel_params[7] = &out_d;
        kernel_params[8] = &arg_d;

        CUDA_SAFE_CALL(cuLaunchKernel(kernel,
                   gridSize.x, gridSize.y, gridSize.z,    // grid dim
                   blockSize.x, blockSize.y, blockSize.z,   // block dim
                   blockSize.x * dimY * sizeof(TYPE), NULL,             // shared mem and stream
                   kernel_params, 0));           // arguments
                   
    } else {
        // simple mode
        
        gridSize.x = nx / blockSize.x + (nx % blockSize.x == 0 ? 0 : 1);
        
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
    }
    
    CUDA_SAFE_CALL(cuCtxSynchronize());

    CUDA_SAFE_CALL(cuModuleUnload(module));
    
    // Send data from device to host.

    if(tagHostDevice==0)
        cudaMemcpy(out, out_d, sizeof(TYPE) * sizeout, cudaMemcpyDeviceToHost);

    cudaFree(p_data);
    if (RR.tagRanges==1) {
        cudaFree(lookup_d);
        cudaFree(slices_x_d);
        cudaFree(ranges_y_d);
        if (SS.nbatchdims > 0)
            cudaFree(offsets_d);
    }

    return 0;
}


extern "C" __host__ int launch_keops_float(const char* ptx_file_name, int tagHostDevice, int dimY, int nx, int ny, int device_id, int tagI, 
                                        int *indsi, int *indsj, int *indsp, 
                                        int dimout, 
                                        int *dimsx, int *dimsy, int *dimsp,
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
                                        indsi, indsj, indsp,
                                        dimout,
                                        dimsx, dimsy, dimsp,
                                        ranges, shapeout, out, nargs, arg, argshape);
                                                                        
}

extern "C" __host__ int launch_keops_double(const char* ptx_file_name, int tagHostDevice, int dimY, int nx, int ny, int device_id, int tagI, 
                                        int *indsi, int *indsj, int *indsp, 
                                        int dimout, 
                                        int *dimsx, int *dimsy, int *dimsp,
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
                                        indsi, indsj, indsp,
                                        dimout,
                                        dimsx, dimsy, dimsp,
                                        ranges, shapeout, out, nargs, arg, argshape);
                                                                        
}
