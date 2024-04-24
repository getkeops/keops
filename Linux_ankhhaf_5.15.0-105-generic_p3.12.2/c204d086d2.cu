
                          
                        #define C_CONTIGUOUS 1
#define USE_HALF 0

                        
                        extern "C" __global__ void GpuConv1DOnDevice(signed long int nx, signed long int ny, float *out, float **arg_4) {
    
                          // get the index of the current thread
                          signed long int i = blockIdx.x * blockDim.x + threadIdx.x;

                          // declare shared mem
                          extern __shared__ float yj[];

                          // load parameters variables from global memory to local thread memory
                          
                          

                          float fout[1];
                          float xi[2];
                          float acc[1];
                          float tmp[1];

                          if (i < nx) {
                             for(int k_41=0; k_41<1; k_41+=(1)) {
                            acc[k_41] = (0.0f);

                        }
                     // acc = 0
                            
                            {
signed long int a=0;

for(signed long int v=0; v<1; v++) {
    xi[a] = arg_4[0][i*1+v];
     a++;
}

for(signed long int v=0; v<1; v++) {
    xi[a] = arg_4[1][i*1+v];
     a++;
}
}
 // load xi variables from global memory to local thread memory
                          }

                          for (signed long int jstart = 0, tile = 0; jstart < ny; jstart += blockDim.x, tile++) {

                            // get the current column
                            signed long int j = tile * blockDim.x + threadIdx.x;

                            if (j < ny) { // we load yj from device global memory only if j<ny
                              {
signed long int a=0;

for(signed long int v=0; v<1; v++) {
    (yj + threadIdx.x * 1)[a] = arg_4[3][j*1+v];
     a++;
}
}
 
                            }
                            __syncthreads();

                            if (i < nx) { // we compute x1i only if needed
                              float * yjrel = yj;
                               for(int k_42=0; k_42<1; k_42+=(1)) {
                            tmp[k_42] = (0.0f);

                        }
                    
                              for (signed long int jrel = 0; (jrel < blockDim.x) && (jrel < ny - jstart); jrel++, yjrel += 1) {
                                
{
// Starting code block for IfElse(Var(0,1,0),1/2*(Var(3,1,1)*Rsqrt(Var(1,1,0))),Zero(1)).

float out_mult_8[1];

{
// Starting code block for 1/2*(Var(3,1,1)*Rsqrt(Var(1,1,0))).

float out_ratcst_1[1];

{
// Starting code block for 1/2.

*out_ratcst_1 = (float)((float)(0.5));


// Finished code block for 1/2.
}

float out_mult_9[1];

{
// Starting code block for Var(3,1,1)*Rsqrt(Var(1,1,0)).

float out_rsqrt_1[1];

{
// Starting code block for Rsqrt(Var(1,1,0)).


 for(int k_43=0; k_43<1; k_43+=(1)) {
                            out_rsqrt_1[(k_43*1)] = (
                    #ifdef __CUDACC__
                        (((xi+1)[(k_43*1)]==0.0f)? 0.0f : rsqrt((xi+1)[(k_43*1)]))
                    #else
                        (((xi+1)[(k_43*1)]==0.0f)? 0.0f : 1.0f/sqrt((xi+1)[(k_43*1)]))
                    #endif
                );

                        }
                    

// Finished code block for Rsqrt(Var(1,1,0)).
}


 for(int k_44=0; k_44<1; k_44+=(1)) {
                            out_mult_9[(k_44*1)] = (
                    #ifdef __CUDACC__
                        ((yjrel+0)[(k_44*1)]*out_rsqrt_1[(k_44*1)])
                    #else
                        ((yjrel+0)[(k_44*1)]*out_rsqrt_1[(k_44*1)])
                    #endif
                );

                        }
                    

// Finished code block for Var(3,1,1)*Rsqrt(Var(1,1,0)).
}


 for(int k_45=0; k_45<1; k_45+=(1)) {
                            out_mult_8[(k_45*1)] = (
                    #ifdef __CUDACC__
                        (out_ratcst_1[(k_45*1)]*out_mult_9[(k_45*1)])
                    #else
                        (out_ratcst_1[(k_45*1)]*out_mult_9[(k_45*1)])
                    #endif
                );

                        }
                    

// Finished code block for 1/2*(Var(3,1,1)*Rsqrt(Var(1,1,0))).
}

float out_zero_2[1];

{
// Starting code block for Zero(1).

 for(int k_46=0; k_46<1; k_46+=(1)) {
                            out_zero_2[k_46] = (0.0f);

                        }
                    

// Finished code block for Zero(1).
}


 for(int k_47=0; k_47<1; k_47+=(1)) {
                            fout[(k_47*1)] = (
                    #ifdef __CUDACC__
                        (((xi+0)[(k_47*1)]>=0.0f) ? out_mult_8[(k_47*1)] : out_zero_2[(k_47*1)])
                    #else
                        (((xi+0)[(k_47*1)]>=0.0f) ? out_mult_8[(k_47*1)] : out_zero_2[(k_47*1)])
                    #endif
                );

                        }
                    

// Finished code block for IfElse(Var(0,1,0),1/2*(Var(3,1,1)*Rsqrt(Var(1,1,0))),Zero(1)).
}

 // Call the function, which outputs results in fout
                                
 for(int k_48=0; k_48<1; k_48+=(1)) {
                            tmp[(k_48*1)] += (fout[(k_48*1)]);

                        }
                    
                              }
                              
 for(int k_49=0; k_49<1; k_49+=(1)) {
                            acc[(k_49*1)] += (tmp[(k_49*1)]);

                        }
                    
                            }
                            __syncthreads();
                          }
                          if (i < nx) {
                            
 for(int k_50=0; k_50<1; k_50+=(1)) {
                            (out + i * 1)[k_50] = (acc[k_50]);

                        }
                     
                          }

                        }
                    