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

template < typename TYPE >
int get_sum(TYPE *shape) {
    int ndims = shape[0];
    int size = 1;
    for (int k=0; k<ndims; k++)
        size *= shape[k+1];
    return size;
}


template < typename TYPE >
void load_args_FromDevice(void*& p_data, TYPE* out, TYPE*& out_d, int nargs, TYPE** arg, TYPE**& arg_d) {
    cudaMalloc(&p_data, sizeof(TYPE*) * nargs);
    out_d = out;
    arg_d = (TYPE **) p_data;
    // copy array of pointers
    cudaMemcpy(arg_d, arg, nargs * sizeof(TYPE *), cudaMemcpyHostToDevice);
}


template < typename TYPE >
void load_args_FromHost(void*& p_data, TYPE* out, TYPE*& out_d, int nargs, TYPE** arg, TYPE**& arg_d, int** argshape, int sizeout) {
    int sizes[nargs];
    int totsize = sizeout;
    for (int k=0; k<nargs; k++) {
        sizes[k] = get_sum(argshape[k]);
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
