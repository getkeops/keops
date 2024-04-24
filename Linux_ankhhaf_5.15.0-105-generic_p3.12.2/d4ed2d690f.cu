
                          
                        #define C_CONTIGUOUS 1
#define USE_HALF 0

                        
                        extern "C" __global__ void GpuConv1DOnDevice(signed long int nx, signed long int ny, float *out, float **arg_2) {
    
                          // get the index of the current thread
                          signed long int i = blockIdx.x * blockDim.x + threadIdx.x;

                          // declare shared mem
                          extern __shared__ float yj[];

                          // load parameters variables from global memory to local thread memory
                          
                          

                          float fout[1];
                          
                          float acc[1];
                          float tmp[1];

                          if (i < nx) {
                             for(int k_24=0; k_24<1; k_24+=(1)) {
                            acc[k_24] = (0.0f);

                        }
                     // acc = 0
                            
                             // load xi variables from global memory to local thread memory
                          }

                          for (signed long int jstart = 0, tile = 0; jstart < ny; jstart += blockDim.x, tile++) {

                            // get the current column
                            signed long int j = tile * blockDim.x + threadIdx.x;

                            if (j < ny) { // we load yj from device global memory only if j<ny
                              {
signed long int a=0;

for(signed long int v=0; v<1; v++) {
    (yj + threadIdx.x * 3)[a] = arg_2[0][j*1+v];
     a++;
}

for(signed long int v=0; v<1; v++) {
    (yj + threadIdx.x * 3)[a] = arg_2[1][j*1+v];
     a++;
}

for(signed long int v=0; v<1; v++) {
    (yj + threadIdx.x * 3)[a] = arg_2[2][j*1+v];
     a++;
}
}
 
                            }
                            __syncthreads();

                            if (i < nx) { // we compute x1i only if needed
                              float * yjrel = yj;
                               for(int k_25=0; k_25<1; k_25+=(1)) {
                            tmp[k_25] = (0.0f);

                        }
                    
                              for (signed long int jrel = 0; (jrel < blockDim.x) && (jrel < ny - jstart); jrel++, yjrel += 3) {
                                
{
// Starting code block for IfElse(Var(0,1,0),Sqrt(Var(1,1,0)),Var(2,1,0)**2*Var(2,1,0)).

float out_sqrt_1[1];

{
// Starting code block for Sqrt(Var(1,1,0)).


 for(int k_26=0; k_26<1; k_26+=(1)) {
                            out_sqrt_1[(k_26*1)] = (
                    #ifdef __CUDACC__
                        sqrt((yjrel+1)[(k_26*1)])
                    #else
                        sqrt((yjrel+1)[(k_26*1)])
                    #endif
                );

                        }
                    

// Finished code block for Sqrt(Var(1,1,0)).
}

float out_mult_7[1];

{
// Starting code block for Var(2,1,0)**2*Var(2,1,0).

float out_square_2[1];

{
// Starting code block for Var(2,1,0)**2.


 for(int k_27=0; k_27<1; k_27+=(1)) {
                            out_square_2[(k_27*1)] = (((yjrel+2)[(k_27*1)]*(yjrel+2)[(k_27*1)]));

                        }
                    

// Finished code block for Var(2,1,0)**2.
}


 for(int k_28=0; k_28<1; k_28+=(1)) {
                            out_mult_7[(k_28*1)] = (
                    #ifdef __CUDACC__
                        (out_square_2[(k_28*1)]*(yjrel+2)[(k_28*1)])
                    #else
                        (out_square_2[(k_28*1)]*(yjrel+2)[(k_28*1)])
                    #endif
                );

                        }
                    

// Finished code block for Var(2,1,0)**2*Var(2,1,0).
}


 for(int k_29=0; k_29<1; k_29+=(1)) {
                            fout[(k_29*1)] = (
                    #ifdef __CUDACC__
                        (((yjrel+0)[(k_29*1)]>=0.0f) ? out_sqrt_1[(k_29*1)] : out_mult_7[(k_29*1)])
                    #else
                        (((yjrel+0)[(k_29*1)]>=0.0f) ? out_sqrt_1[(k_29*1)] : out_mult_7[(k_29*1)])
                    #endif
                );

                        }
                    

// Finished code block for IfElse(Var(0,1,0),Sqrt(Var(1,1,0)),Var(2,1,0)**2*Var(2,1,0)).
}

 // Call the function, which outputs results in fout
                                
 for(int k_30=0; k_30<1; k_30+=(1)) {
                            tmp[(k_30*1)] += (fout[(k_30*1)]);

                        }
                    
                              }
                              
 for(int k_31=0; k_31<1; k_31+=(1)) {
                            acc[(k_31*1)] += (tmp[(k_31*1)]);

                        }
                    
                            }
                            __syncthreads();
                          }
                          if (i < nx) {
                            
 for(int k_32=0; k_32<1; k_32+=(1)) {
                            (out + i * 1)[k_32] = (acc[k_32]);

                        }
                     
                          }

                        }
                    