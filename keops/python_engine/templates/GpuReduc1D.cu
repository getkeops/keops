#define DIRECT_SUM 0
#define BLOCK_SUM 1
#define KAHAN_SCHEME 2

#ifndef USE_HALF
  #define USE_HALF 0
#endif

#if USE_HALF
  #include <cuda_fp16.h>
#endif

__global__ void GpuConv1DOnDevice(int nx, int ny, {dtype} *out, {signature_list(args)}) {{
    
  // get the index of the current thread
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // declare shared mem
  extern __shared__ {dtype} yj[];

  // load parameters variables from global memory to local thread memory
  {param_loc.declare()}
  {varloader.load_vars("p", param_loc, args)}

  {fout.declare()}
  {xi.declare()}
  {acc.declare()}
#if SUM_SCHEME == BLOCK_SUM
  {tmp.declare()}
#elif SUM_SCHEME == KAHAN_SCHEME
  {tmp_kahan.declare()}
#endif
	  
  if (i < nx) {{
    {red_formula.InitializeReduction(acc)} // acc = 0
#if SUM_SCHEME == KAHAN_SCHEME
    {tmp_kahan.assign(c_zero_float)}
#endif
    {varloader.load_vars('i', xi, args, row_index=i)} // load xi variables from global memory to local thread memory
  }}

  for (int jstart = 0, tile = 0; jstart < ny; jstart += blockDim.x, tile++) {{

    // get the current column
    int j = tile * blockDim.x + threadIdx.x;

    if (j < ny) {{ // we load yj from device global memory only if j<ny
      {varloader.load_vars("j", yjloc, args, row_index=j)} 
    }}
    __syncthreads();

    if (i < nx) {{ // we compute x1i only if needed
      {dtype} * yjrel = yj;
#if SUM_SCHEME == BLOCK_SUM
      {red_formula.InitializeReduction(tmp)}
#endif
      for (int jrel = 0; (jrel < blockDim.x) && (jrel < ny - jstart); jrel++, yjrel += {varloader.dimy}) {{
        {red_formula.formula(fout,table)} // Call the function, which outputs results in fout
#if SUM_SCHEME == BLOCK_SUM
    #if USE_HALF
        int ind = jrel + tile * blockDim.x;
        {red_formula.ReducePairShort(tmp, fout, ind_pack_half2)}     // tmp += fout
    #else
        {red_formula.ReducePairShort(tmp, fout, jreltile)}  // tmp += fout
    #endif
#elif SUM_SCHEME == KAHAN_SCHEME
        {red_formula.KahanScheme(acc, fout, tmp_kahan)}
#else
    #if USE_HALF
        int ind = jrel + tile * blockDim.x;
        {red_formula.ReducePairShort(acc, fout, ind_pack_half2)}       // acc += fout
    #else
	    {red_formula.ReducePairShort(acc, fout, jreltile)} // acc += fout
    #endif
#endif
      }}
#if SUM_SCHEME == BLOCK_SUM
      {red_formula.ReducePair(acc, tmp)}  // acc += tmp
#endif
    }}
    __syncthreads();
  }}
  if (i < nx) {{
    {red_formula.FinalizeOutput(acc, outi, i)} 
  }}

}}





extern "C" __host__ int Eval(int nx, int ny, int device_id, {dtype} *out, {signature_list(args)}) {{

    // device_id is provided, so we set the GPU device accordingly
    // Warning : is has to be consistent with location of data
    cudaSetDevice(device_id);
	
    // Compute on device : grid and block are both 1d

    //SetGpuProps(devise_id);

    dim3 blockSize;

    blockSize.x = 32;
	
    dim3 gridSize;
    gridSize.x = nx / blockSize.x + (nx % blockSize.x == 0 ? 0 : 1);

    GpuConv1DOnDevice <<< gridSize, blockSize, blockSize.x * {varloader.dimy} * sizeof({dtype}) >>> (nx, ny, out, {call_list(args)});
    
    // block until the device has completed
    cudaDeviceSynchronize();

    //CudaCheckError();

    return 0;
}}

