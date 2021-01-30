
extern "C" __global__ void GpuConv1DOnDevice(int nx, int ny, {TYPE} *out {args}) {{

  {TYPE}* args[{nargs}];
  {loadargs}
    
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
