
                        #define C_CONTIGUOUS 1
#define USE_HALF 0


                        extern "C" __global__ void GpuConv1DOnDevice_ranges(signed long int nx, signed long int ny, int nbatchdims,
                                                    signed long int *offsets_d, signed long int *lookup_d, signed long int *slices_x,
                                                    signed long int *ranges_y, float *out, float **arg5, signed long int nx_org, 
                                                    signed long int ny_org) {
                                                        
                          signed long int offsets[3];
                          signed long int *indices_i = offsets;
                          signed long int *indices_j = offsets + 1;
                          
                          
                          if (nbatchdims > 0)
                              for (int k = 0; k < 3; k++)
                                  offsets[k] = offsets_d[ 3 * blockIdx.x + k ];
                          // Retrieve our position along the laaaaarge [1,~nx] axis: -----------------
                          signed long int range_id= (lookup_d)[3*blockIdx.x] ;
                          signed long int start_x = (lookup_d)[3*blockIdx.x+1] ;
                          signed long int end_x   = (lookup_d)[3*blockIdx.x+2] ;
  
                          // The "slices_x" vector encodes a set of cutting points in
                          // the "ranges_y" array of ranges.
                          // As discussed in the Genred docstring, the first "0" is implicit:
                          signed long int start_slice = range_id < 1 ? 0 : slices_x[range_id-1];
                          signed long int end_slice   = slices_x[range_id];

                          // get the index of the current thread
                          signed long int i = start_x + threadIdx.x;

                          // declare shared mem
                          extern __shared__ float yj[];
                          
                          // load parameter(s)
                          

                          if (nbatchdims == 0) {
                              
                          } else {
                              
                          }
                          float fout[1];
                          float xi[1];
                          float acc[1];
                          float tmp[1];

                          if(i<end_x) {
                              acc[0] = 0.0f; // acc = 0
                              
                              if (nbatchdims == 0) {
                                  {
    signed long int a = (signed long int)0;
    for(signed long int v = (signed long int)0; v<1; v++)
    {
        xi[a] = (arg5[0])[(i+v)];
        a++;
    }
} // load xi variables from global memory to local thread memory
                              } else {
                                  {
    signed long int a = (signed long int)0;
    for(signed long int v = (signed long int)0; v<1; v++)
    {
        xi[a] = (arg5[0])[((threadIdx.x+indices_i[0])+v)];
        a++;
    }
}  // Possibly, with offsets as we support broadcasting over batch dimensions
                              }
                          }
                          
                          signed long int start_y = ranges_y[2*start_slice], end_y = 0;
                          for( signed long int index = start_slice ; index < end_slice ; index++ ) {
                              if( (index+1 >= end_slice) || (ranges_y[2*index+2] != ranges_y[2*index+1]) ) {
                                  //start_y = ranges_y[2*index] ;
                                  end_y = ranges_y[2*index+1];

                                  for(signed long int jstart = start_y, tile = 0; jstart < end_y; jstart += blockDim.x, tile++) {
                                      // get the current column
                                      signed long int j = jstart + threadIdx.x;

                                      if(j<end_y) // we load yj from device global memory only if j<end_y
                                          if (nbatchdims == 0) {
                                              {
    signed long int a = (signed long int)0;
    for(signed long int v = (signed long int)0; v<1; v++)
    {
        (yj + threadIdx.x * 2)[a] = (arg5[1])[(j+v)];
        a++;
    }
    for(signed long int v = (signed long int)0; v<1; v++)
    {
        (yj + threadIdx.x * 2)[a] = (arg5[2])[(j+v)];
        a++;
    }
} // load yj variables from global memory to shared memory
                                          } else {
                                              {
    signed long int a = (signed long int)0;
    for(signed long int v = (signed long int)0; v<1; v++)
    {
        (yj + threadIdx.x * 2)[a] = (arg5[1])[(((j-start_y)+indices_j[0])+v)];
        a++;
    }
    for(signed long int v = (signed long int)0; v<1; v++)
    {
        (yj + threadIdx.x * 2)[a] = (arg5[2])[(((j-start_y)+indices_j[1])+v)];
        a++;
    }
}  // Possibly, with offsets as we support broadcasting over batch dimensions
                                          }
                                      __syncthreads();
                                      
                                      if(i<end_x) { // we compute x1i only if needed
                                          float * yjrel = yj; // Loop on the columns of the current block.
                                          tmp[0] = 0.0f;
                                          if (nbatchdims == 0) {
                                              for(signed long int jrel = 0; (jrel < blockDim.x) && (jrel<end_y-jstart); jrel++, yjrel+=2) {
                                                  // Starting code block for Var(2,1,1)*Exp(-1/1501*I()-1/2000*(J()*(Var(0,1,0)-Var(1,1,1))**2));
float out_square6;
out_square6 = (xi+0)[0]-(yjrel+0)[0];
out_square6 *= out_square6;
fout[0] = (yjrel+1)[0]*(exp((((-1/1501)*(float)i)-((1/2000)*((float)((jrel + tile * blockDim.x)+start_y)*out_square6)))));
// Finished code block for Var(2,1,1)*Exp(-1/1501*I()-1/2000*(J()*(Var(0,1,0)-Var(1,1,1))**2)); // Call the function, which outputs results in xi[0:DIMX1]
                                                  tmp[0] += fout[0];
                                              } 
                                          } else {
                                              for(signed long int jrel = 0; (jrel < blockDim.x) && (jrel<end_y-jstart); jrel++, yjrel+=2) {
                                                  // Starting code block for Var(2,1,1)*Exp(-1/1501*I()-1/2000*(J()*(Var(0,1,0)-Var(1,1,1))**2));
float out_square7;
out_square7 = (xi+0)[0]-(yjrel+0)[0];
out_square7 *= out_square7;
fout[0] = (yjrel+1)[0]*(exp((((-1/1501)*(float)(i%nx_org))-((1/2000)*((float)(jrel + tile * blockDim.x)*out_square7)))));
// Finished code block for Var(2,1,1)*Exp(-1/1501*I()-1/2000*(J()*(Var(0,1,0)-Var(1,1,1))**2)); // Call the function, which outputs results in fout
                                                  tmp[0] += fout[0];
                                              }
                                          }
                                          acc[0] += tmp[0];
                                      }
                                      __syncthreads();
                                  }
                                  if(index+1 < end_slice)
                                      start_y = ranges_y[2*index+2] ;
                              }
                          }
                          if(i<end_x) {
                          	(out + i * 1)[0] = acc[0]; 
                          }
                      }
                    