
                          
                        #define C_CONTIGUOUS 1
#define USE_HALF 0

                        
                        extern "C" __global__ void GpuConv1DOnDevice(signed long int nx, signed long int ny, float *out, float **arg_1) {
    
                          // get the index of the current thread
                          signed long int i = blockIdx.x * blockDim.x + threadIdx.x;

                          // declare shared mem
                          extern __shared__ float yj[];

                          // load parameters variables from global memory to local thread memory
                          
                          

                          float fout[1];
                          float xi[1];
                          float acc[1];
                          float tmp[1];

                          if (i < nx) {
                             for(int k_9=0; k_9<1; k_9+=(1)) {
                            acc[k_9] = (0.0f);

                        }
                     // acc = 0
                            
                            {
signed long int a=0;

for(signed long int v=0; v<1; v++) {
    xi[a] = arg_1[0][i*1+v];
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
    (yj + threadIdx.x * 1)[a] = arg_1[1][j*1+v];
     a++;
}
}
 
                            }
                            __syncthreads();

                            if (i < nx) { // we compute x1i only if needed
                              float * yjrel = yj;
                               for(int k_10=0; k_10<1; k_10+=(1)) {
                            tmp[k_10] = (0.0f);

                        }
                    
                              for (signed long int jrel = 0; (jrel < blockDim.x) && (jrel < ny - jstart); jrel++, yjrel += 1) {
                                
{
// Starting code block for IfElse(Var(0,1,0),1/2*(Var(1,1,1)*Rsqrt(Var(0,1,0))),2*((Var(0,1,0)*Var(1,1,1))*Var(0,1,0))+Var(0,1,0)**2*Var(1,1,1)).

float out_mult_1[1];

{
// Starting code block for 1/2*(Var(1,1,1)*Rsqrt(Var(0,1,0))).

float out_ratcst_0[1];

{
// Starting code block for 1/2.

*out_ratcst_0 = (float)((float)(0.5));


// Finished code block for 1/2.
}

float out_mult_2[1];

{
// Starting code block for Var(1,1,1)*Rsqrt(Var(0,1,0)).

float out_rsqrt_0[1];

{
// Starting code block for Rsqrt(Var(0,1,0)).


 for(int k_11=0; k_11<1; k_11+=(1)) {
                            out_rsqrt_0[(k_11*1)] = (
                    #ifdef __CUDACC__
                        (((xi+0)[(k_11*1)]==0.0f)? 0.0f : rsqrt((xi+0)[(k_11*1)]))
                    #else
                        (((xi+0)[(k_11*1)]==0.0f)? 0.0f : 1.0f/sqrt((xi+0)[(k_11*1)]))
                    #endif
                );

                        }
                    

// Finished code block for Rsqrt(Var(0,1,0)).
}


 for(int k_12=0; k_12<1; k_12+=(1)) {
                            out_mult_2[(k_12*1)] = (
                    #ifdef __CUDACC__
                        ((yjrel+0)[(k_12*1)]*out_rsqrt_0[(k_12*1)])
                    #else
                        ((yjrel+0)[(k_12*1)]*out_rsqrt_0[(k_12*1)])
                    #endif
                );

                        }
                    

// Finished code block for Var(1,1,1)*Rsqrt(Var(0,1,0)).
}


 for(int k_13=0; k_13<1; k_13+=(1)) {
                            out_mult_1[(k_13*1)] = (
                    #ifdef __CUDACC__
                        (out_ratcst_0[(k_13*1)]*out_mult_2[(k_13*1)])
                    #else
                        (out_ratcst_0[(k_13*1)]*out_mult_2[(k_13*1)])
                    #endif
                );

                        }
                    

// Finished code block for 1/2*(Var(1,1,1)*Rsqrt(Var(0,1,0))).
}

float out_add_0[1];

{
// Starting code block for 2*((Var(0,1,0)*Var(1,1,1))*Var(0,1,0))+Var(0,1,0)**2*Var(1,1,1).

float out_mult_3[1];

{
// Starting code block for 2*((Var(0,1,0)*Var(1,1,1))*Var(0,1,0)).

float out_intcst_0[1];

{
// Starting code block for 2.

*out_intcst_0 = (float)((float)2);


// Finished code block for 2.
}

float out_mult_4[1];

{
// Starting code block for (Var(0,1,0)*Var(1,1,1))*Var(0,1,0).

float out_mult_5[1];

{
// Starting code block for Var(0,1,0)*Var(1,1,1).


 for(int k_14=0; k_14<1; k_14+=(1)) {
                            out_mult_5[(k_14*1)] = (
                    #ifdef __CUDACC__
                        ((xi+0)[(k_14*1)]*(yjrel+0)[(k_14*1)])
                    #else
                        ((xi+0)[(k_14*1)]*(yjrel+0)[(k_14*1)])
                    #endif
                );

                        }
                    

// Finished code block for Var(0,1,0)*Var(1,1,1).
}


 for(int k_15=0; k_15<1; k_15+=(1)) {
                            out_mult_4[(k_15*1)] = (
                    #ifdef __CUDACC__
                        (out_mult_5[(k_15*1)]*(xi+0)[(k_15*1)])
                    #else
                        (out_mult_5[(k_15*1)]*(xi+0)[(k_15*1)])
                    #endif
                );

                        }
                    

// Finished code block for (Var(0,1,0)*Var(1,1,1))*Var(0,1,0).
}


 for(int k_16=0; k_16<1; k_16+=(1)) {
                            out_mult_3[(k_16*1)] = (
                    #ifdef __CUDACC__
                        (out_intcst_0[(k_16*1)]*out_mult_4[(k_16*1)])
                    #else
                        (out_intcst_0[(k_16*1)]*out_mult_4[(k_16*1)])
                    #endif
                );

                        }
                    

// Finished code block for 2*((Var(0,1,0)*Var(1,1,1))*Var(0,1,0)).
}

float out_mult_6[1];

{
// Starting code block for Var(0,1,0)**2*Var(1,1,1).

float out_square_1[1];

{
// Starting code block for Var(0,1,0)**2.


 for(int k_17=0; k_17<1; k_17+=(1)) {
                            out_square_1[(k_17*1)] = (((xi+0)[(k_17*1)]*(xi+0)[(k_17*1)]));

                        }
                    

// Finished code block for Var(0,1,0)**2.
}


 for(int k_18=0; k_18<1; k_18+=(1)) {
                            out_mult_6[(k_18*1)] = (
                    #ifdef __CUDACC__
                        (out_square_1[(k_18*1)]*(yjrel+0)[(k_18*1)])
                    #else
                        (out_square_1[(k_18*1)]*(yjrel+0)[(k_18*1)])
                    #endif
                );

                        }
                    

// Finished code block for Var(0,1,0)**2*Var(1,1,1).
}


 for(int k_19=0; k_19<1; k_19+=(1)) {
                            out_add_0[(k_19*1)] = out_mult_3[(k_19*1)]+out_mult_6[(k_19*1)];

                        }
                    

// Finished code block for 2*((Var(0,1,0)*Var(1,1,1))*Var(0,1,0))+Var(0,1,0)**2*Var(1,1,1).
}


 for(int k_20=0; k_20<1; k_20+=(1)) {
                            fout[(k_20*1)] = (
                    #ifdef __CUDACC__
                        (((xi+0)[(k_20*1)]>=0.0f) ? out_mult_1[(k_20*1)] : out_add_0[(k_20*1)])
                    #else
                        (((xi+0)[(k_20*1)]>=0.0f) ? out_mult_1[(k_20*1)] : out_add_0[(k_20*1)])
                    #endif
                );

                        }
                    

// Finished code block for IfElse(Var(0,1,0),1/2*(Var(1,1,1)*Rsqrt(Var(0,1,0))),2*((Var(0,1,0)*Var(1,1,1))*Var(0,1,0))+Var(0,1,0)**2*Var(1,1,1)).
}

 // Call the function, which outputs results in fout
                                
 for(int k_21=0; k_21<1; k_21+=(1)) {
                            tmp[(k_21*1)] += (fout[(k_21*1)]);

                        }
                    
                              }
                              
 for(int k_22=0; k_22<1; k_22+=(1)) {
                            acc[(k_22*1)] += (tmp[(k_22*1)]);

                        }
                    
                            }
                            __syncthreads();
                          }
                          if (i < nx) {
                            
 for(int k_23=0; k_23<1; k_23+=(1)) {
                            (out + i * 1)[k_23] = (acc[k_23]);

                        }
                     
                          }

                        }
                    