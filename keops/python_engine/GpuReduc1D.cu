
__global__ void GpuConv1DOnDevice(int nx, int ny, {TYPE} *out, {TYPE} **args) {{

  // get the index of the current thread
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // declare shared mem
  extern __shared__ {TYPE} yj[];

  // load parameters variables from global memory to local thread memory
  {definep}
  {loadp}

  {TYPE} fout[{DIMFOUT}];
  {definex}
  {TYPEACC} acc[{DIMRED}];
  
  if (i < nx) {{
    {InitializeReduction} // acc = 0
    {loadx} // load xi variables from global memory to local thread memory
  }}

  for (int jstart = 0, tile = 0; jstart < ny; jstart += blockDim.x, tile++) {{

    // get the current column
    int j = tile * blockDim.x + threadIdx.x;

    if (j < ny) {{ // we load yj from device global memory only if j<ny
      {loady} 
    }}
    __syncthreads();

    if (i < nx) {{ // we compute x1i only if needed
      {TYPE} * yjrel = yj; // Loop on the columns of the current block.
      for (int jrel = 0; (jrel < blockDim.x) && (jrel < ny - jstart); jrel++, yjrel += {DIMY}) {{
        {call} // Call the function, which outputs results in fout
	    {ReducePairShort} // acc += fout
      }}
    }}
    __syncthreads();
  }}
  if (i < nx) {{
    {FinalizeOutput} 
  }}

}}





  extern "C" __host__ int Eval(int nx, int ny, {TYPE} *out {args}) {{

	{TYPE}* args[{nargs}];
	{loadargs}
		  
    // device array of pointers to device data
    {TYPE} **args_d;

    // single cudaMalloc
    cudaMalloc(&args_d, sizeof({TYPE} *) * {NMINARGS});

    cudaMemcpy(args_d, args, {NMINARGS} * sizeof({TYPE} *), cudaMemcpyHostToDevice);

    // Compute on device : grid and block are both 1d

    //int dev = -1;
    //cudaGetDevice(&dev);

    //SetGpuProps(dev);

    dim3 blockSize;

    blockSize.x = 32;
	
    dim3 gridSize;
    gridSize.x = nx / blockSize.x + (nx % blockSize.x == 0 ? 0 : 1);

    GpuConv1DOnDevice <<< gridSize, blockSize, blockSize.x * {DIMY} * sizeof({TYPE}) >>> (nx, ny, out, args_d);
    
    // block until the device has completed
    cudaDeviceSynchronize();

    //CudaCheckError();

    cudaFree(args_d);

    return 0;
  }}
