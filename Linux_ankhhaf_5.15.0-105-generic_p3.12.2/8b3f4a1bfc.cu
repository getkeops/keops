
                          
                        #define C_CONTIGUOUS 1
#define USE_HALF 0

                        
                        extern "C" __global__ void GpuConv1DOnDevice(signed long int nx, signed long int ny, float *out, float **arg_0) {
    
                          // get the index of the current thread
                          signed long int i = blockIdx.x * blockDim.x + threadIdx.x;

                          // declare shared mem
                          extern __shared__ float yj[];

                          // load parameters variables from global memory to local thread memory
                          
                          

                          float fout[1];
                          
                          float acc[1];
                          float tmp[1];

                          if (i < nx) {
                             for(int k_0=0; k_0<1; k_0+=(1)) {
                            acc[k_0] = (0.0f);

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
    (yj + threadIdx.x * 1)[a] = arg_0[0][j*1+v];
     a++;
}
}
 
                            }
                            __syncthreads();

                            if (i < nx) { // we compute x1i only if needed
                              float * yjrel = yj;
                               for(int k_1=0; k_1<1; k_1+=(1)) {
                            tmp[k_1] = (0.0f);

                        }
                    
                              for (signed long int jrel = 0; (jrel < blockDim.x) && (jrel < ny - jstart); jrel++, yjrel += 1) {
                                
{
// Starting code block for IfElse(Var(0,1,0),Sqrt(Var(0,1,0)),Var(0,1,0)**2*Var(0,1,0)).

float out_sqrt_0[1];

{
// Starting code block for Sqrt(Var(0,1,0)).


 for(int k_2=0; k_2<1; k_2+=(1)) {
                            out_sqrt_0[(k_2*1)] = (
                    #ifdef __CUDACC__
                        sqrt((yjrel+0)[(k_2*1)])
                    #else
                        sqrt((yjrel+0)[(k_2*1)])
                    #endif
                );

                        }
                    

// Finished code block for Sqrt(Var(0,1,0)).
}

float out_mult_0[1];

{
// Starting code block for Var(0,1,0)**2*Var(0,1,0).

float out_square_0[1];

{
// Starting code block for Var(0,1,0)**2.


 for(int k_3=0; k_3<1; k_3+=(1)) {
                            out_square_0[(k_3*1)] = (((yjrel+0)[(k_3*1)]*(yjrel+0)[(k_3*1)]));

                        }
                    

// Finished code block for Var(0,1,0)**2.
}


 for(int k_4=0; k_4<1; k_4+=(1)) {
                            out_mult_0[(k_4*1)] = (
                    #ifdef __CUDACC__
                        (out_square_0[(k_4*1)]*(yjrel+0)[(k_4*1)])
                    #else
                        (out_square_0[(k_4*1)]*(yjrel+0)[(k_4*1)])
                    #endif
                );

                        }
                    

// Finished code block for Var(0,1,0)**2*Var(0,1,0).
}


 for(int k_5=0; k_5<1; k_5+=(1)) {
                            fout[(k_5*1)] = (
                    #ifdef __CUDACC__
                        (((yjrel+0)[(k_5*1)]>=0.0f) ? out_sqrt_0[(k_5*1)] : out_mult_0[(k_5*1)])
                    #else
                        (((yjrel+0)[(k_5*1)]>=0.0f) ? out_sqrt_0[(k_5*1)] : out_mult_0[(k_5*1)])
                    #endif
                );

                        }
                    

// Finished code block for IfElse(Var(0,1,0),Sqrt(Var(0,1,0)),Var(0,1,0)**2*Var(0,1,0)).
}

 // Call the function, which outputs results in fout
                                
 for(int k_6=0; k_6<1; k_6+=(1)) {
                            tmp[(k_6*1)] += (fout[(k_6*1)]);

                        }
                    
                              }
                              
 for(int k_7=0; k_7<1; k_7+=(1)) {
                            acc[(k_7*1)] += (tmp[(k_7*1)]);

                        }
                    
                            }
                            __syncthreads();
                          }
                          if (i < nx) {
                            
 for(int k_8=0; k_8<1; k_8+=(1)) {
                            (out + i * 1)[k_8] = (acc[k_8]);

                        }
                     
                          }

                        }
                    