
__global__ void GpuConv1DOnDevice(int nx, int ny, float *out, float **args) {

  // get the index of the current thread
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // declare shared mem
  extern __shared__ float yj[];

  // load parameter(s)
  float param_loc[0 < 1 ? 1 : 0];
  loadp //load<DIMSP, INDSP>(0, param_loc, args); // load parameters variables from global memory to local thread memory

  float fout[1];
  // get the value of variable (index with i)
  float xi[2 < 1 ? 1 : 2];
  float acc[1];
  
  if (i < nx) {
    #pragma unroll
for(int k=0; k<1; k++)
    acc[k] = (float)(0.0f); //<__TYPEACC__, TYPE >()(acc); // acc = 0
    xi[0] = args[0][i*2+0];
xi[1] = args[0][i*2+1];
 //<DIMSX, INDSI>(i, xi, args); // load xi variables from global memory to local thread memory
  }

  for (int jstart = 0, tile = 0; jstart < ny; jstart += blockDim.x, tile++) {

    // get the current column
    int j = tile * blockDim.x + threadIdx.x;

    if (j < ny) { // we load yj from device global memory only if j<ny
      yj + threadIdx.x * 2[0] = args[1][j*2+0];
yj + threadIdx.x * 2[1] = args[1][j*2+1];
 //<DIMSY,INDSJ>(j, yj + threadIdx.x * DIMY, args); // load yj variables from global memory to shared memory
    }
    __syncthreads();

    if (i < nx) { // we compute x1i only if needed
      float * yjrel = yj; // Loop on the columns of the current block.
      for (int jrel = 0; (jrel < blockDim.x) && (jrel < ny - jstart); jrel++, yjrel += 2) {
        
{
// Starting code block for Exp(-Sum((Var(0,2,0)-Var(1,2,1))**2)).

float out_minus_1[1];

{
// Starting code block for -Sum((Var(0,2,0)-Var(1,2,1))**2).

float out_sum_1[1];

{
// Starting code block for Sum((Var(0,2,0)-Var(1,2,1))**2).

float out_square_1[2];

{
// Starting code block for (Var(0,2,0)-Var(1,2,1))**2.

float out_subtract_1[2];

{
// Starting code block for Var(0,2,0)-Var(1,2,1).

#pragma unroll
for(int k=0; k<2; k++) {
    out_subtract_1[k*1] = (xi+0)[k*1]-(yj + threadIdx.x * 2+0)[k*1];
 }


// Finished code block for Var(0,2,0)-Var(1,2,1).
}

#pragma unroll
for(int k=0; k<2; k++) {
    out_square_1[k*1] = out_subtract_1[k*1]*out_subtract_1[k*1];
 }


// Finished code block for (Var(0,2,0)-Var(1,2,1))**2.
}

*out_sum_1 = (float)(0.0f);
#pragma unroll
for(int k=0; k<2; k++) {
    out_sum_1[k*0] += out_square_1[k*1];
 }


// Finished code block for Sum((Var(0,2,0)-Var(1,2,1))**2).
}

#pragma unroll
for(int k=0; k<1; k++) {
    out_minus_1[k*1] = -out_sum_1[k*1];
 }


// Finished code block for -Sum((Var(0,2,0)-Var(1,2,1))**2).
}

#pragma unroll
for(int k=0; k<1; k++) {
    fout[k*1] = exp(out_minus_1[k*1]);
 }


// Finished code block for Exp(-Sum((Var(0,2,0)-Var(1,2,1))**2)).
}

 //<DIMSX, DIMSY, DIMSP>(fun,fout,xi,yjrel,param_loc); // Call the function, which outputs results in fout
	    #pragma unroll
for(int k=0; k<1; k++) {
    acc[k*1] += (float)(fout[k*1]); }
 //<__TYPEACC__,TYPE>()(acc, fout, jrel + tile * blockDim.x);     // acc += fout
      }
    }
    __syncthreads();
  }
  if (i < nx) {
    #pragma unroll
for(int k=0; k<1; k++)
    (out + i * 1)[k] = (float)acc[k];
 //<__TYPEACC__,TYPE>()(acc, out + i * DIMOUT, i);
  }

}





  extern int GpuConv1D_FromDevice(int nx, int ny, float *out , float* arg0, float* arg1, float* arg2, float* arg3, float* arg4, float* arg5) {

	float* args[6];
	args[0] = arg0;
args[1] = arg1;
args[2] = arg2;
args[3] = arg3;
args[4] = arg4;
args[5] = arg5;

		  
    // device array of pointers to device data
    float **args_d;

    // single cudaMalloc
    CudaSafeCall(cudaMalloc(&args_d, sizeof(float *) * 2));

    CudaSafeCall(cudaMemcpy(args_d, args, 2 * sizeof(float *), cudaMemcpyHostToDevice));

    // Compute on device : grid and block are both 1d

    int dev = -1;
    CudaSafeCall(cudaGetDevice(&dev));

    SetGpuProps(dev);

    dim3 blockSize;

    blockSize.x = 192;
	
    dim3 gridSize;
    gridSize.x = nx / blockSize.x + (nx % blockSize.x == 0 ? 0 : 1);

    GpuConv1DOnDevice <<< gridSize, blockSize, blockSize.x * 2 * sizeof(float) >>> (nx, ny, out, args_d);
    
    // block until the device has completed
    CudaSafeCall(cudaDeviceSynchronize());

    CudaCheckError();

    CudaSafeCall(cudaFree(args_d));

    return 0;
  }
