
                          
                        #define C_CONTIGUOUS 1
#define USE_HALF 0

                        
                        extern "C" __global__ void GpuConv1DOnDevice(signed long int nx, signed long int ny, float *out, float **arg_5) {
    
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
                             for(int k_51=0; k_51<1; k_51+=(1)) {
                            acc[k_51] = (0.0f);

                        }
                     // acc = 0
                            
                            {
signed long int a=0;

for(signed long int v=0; v<1; v++) {
    xi[a] = arg_5[0][i*1+v];
     a++;
}

for(signed long int v=0; v<1; v++) {
    xi[a] = arg_5[2][i*1+v];
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
    (yj + threadIdx.x * 1)[a] = arg_5[3][j*1+v];
     a++;
}
}
 
                            }
                            __syncthreads();

                            if (i < nx) { // we compute x1i only if needed
                              float * yjrel = yj;
                               for(int k_52=0; k_52<1; k_52+=(1)) {
                            tmp[k_52] = (0.0f);

                        }
                    
                              for (signed long int jrel = 0; (jrel < blockDim.x) && (jrel < ny - jstart); jrel++, yjrel += 1) {
                                
{
// Starting code block for IfElse(Var(0,1,0),Zero(1),2*((Var(2,1,0)*Var(3,1,1))*Var(2,1,0))+Var(2,1,0)**2*Var(3,1,1)).

float out_zero_3[1];

{
// Starting code block for Zero(1).

 for(int k_53=0; k_53<1; k_53+=(1)) {
                            out_zero_3[k_53] = (0.0f);

                        }
                    

// Finished code block for Zero(1).
}

float out_add_1[1];

{
// Starting code block for 2*((Var(2,1,0)*Var(3,1,1))*Var(2,1,0))+Var(2,1,0)**2*Var(3,1,1).

float out_mult_10[1];

{
// Starting code block for 2*((Var(2,1,0)*Var(3,1,1))*Var(2,1,0)).

float out_intcst_1[1];

{
// Starting code block for 2.

*out_intcst_1 = (float)((float)2);


// Finished code block for 2.
}

float out_mult_11[1];

{
// Starting code block for (Var(2,1,0)*Var(3,1,1))*Var(2,1,0).

float out_mult_12[1];

{
// Starting code block for Var(2,1,0)*Var(3,1,1).


 for(int k_54=0; k_54<1; k_54+=(1)) {
                            out_mult_12[(k_54*1)] = (
                    #ifdef __CUDACC__
                        ((xi+1)[(k_54*1)]*(yjrel+0)[(k_54*1)])
                    #else
                        ((xi+1)[(k_54*1)]*(yjrel+0)[(k_54*1)])
                    #endif
                );

                        }
                    

// Finished code block for Var(2,1,0)*Var(3,1,1).
}


 for(int k_55=0; k_55<1; k_55+=(1)) {
                            out_mult_11[(k_55*1)] = (
                    #ifdef __CUDACC__
                        (out_mult_12[(k_55*1)]*(xi+1)[(k_55*1)])
                    #else
                        (out_mult_12[(k_55*1)]*(xi+1)[(k_55*1)])
                    #endif
                );

                        }
                    

// Finished code block for (Var(2,1,0)*Var(3,1,1))*Var(2,1,0).
}


 for(int k_56=0; k_56<1; k_56+=(1)) {
                            out_mult_10[(k_56*1)] = (
                    #ifdef __CUDACC__
                        (out_intcst_1[(k_56*1)]*out_mult_11[(k_56*1)])
                    #else
                        (out_intcst_1[(k_56*1)]*out_mult_11[(k_56*1)])
                    #endif
                );

                        }
                    

// Finished code block for 2*((Var(2,1,0)*Var(3,1,1))*Var(2,1,0)).
}

float out_mult_13[1];

{
// Starting code block for Var(2,1,0)**2*Var(3,1,1).

float out_square_3[1];

{
// Starting code block for Var(2,1,0)**2.


 for(int k_57=0; k_57<1; k_57+=(1)) {
                            out_square_3[(k_57*1)] = (((xi+1)[(k_57*1)]*(xi+1)[(k_57*1)]));

                        }
                    

// Finished code block for Var(2,1,0)**2.
}


 for(int k_58=0; k_58<1; k_58+=(1)) {
                            out_mult_13[(k_58*1)] = (
                    #ifdef __CUDACC__
                        (out_square_3[(k_58*1)]*(yjrel+0)[(k_58*1)])
                    #else
                        (out_square_3[(k_58*1)]*(yjrel+0)[(k_58*1)])
                    #endif
                );

                        }
                    

// Finished code block for Var(2,1,0)**2*Var(3,1,1).
}


 for(int k_59=0; k_59<1; k_59+=(1)) {
                            out_add_1[(k_59*1)] = out_mult_10[(k_59*1)]+out_mult_13[(k_59*1)];

                        }
                    

// Finished code block for 2*((Var(2,1,0)*Var(3,1,1))*Var(2,1,0))+Var(2,1,0)**2*Var(3,1,1).
}


 for(int k_60=0; k_60<1; k_60+=(1)) {
                            fout[(k_60*1)] = (
                    #ifdef __CUDACC__
                        (((xi+0)[(k_60*1)]>=0.0f) ? out_zero_3[(k_60*1)] : out_add_1[(k_60*1)])
                    #else
                        (((xi+0)[(k_60*1)]>=0.0f) ? out_zero_3[(k_60*1)] : out_add_1[(k_60*1)])
                    #endif
                );

                        }
                    

// Finished code block for IfElse(Var(0,1,0),Zero(1),2*((Var(2,1,0)*Var(3,1,1))*Var(2,1,0))+Var(2,1,0)**2*Var(3,1,1)).
}

 // Call the function, which outputs results in fout
                                
 for(int k_61=0; k_61<1; k_61+=(1)) {
                            tmp[(k_61*1)] += (fout[(k_61*1)]);

                        }
                    
                              }
                              
 for(int k_62=0; k_62<1; k_62+=(1)) {
                            acc[(k_62*1)] += (tmp[(k_62*1)]);

                        }
                    
                            }
                            __syncthreads();
                          }
                          if (i < nx) {
                            
 for(int k_63=0; k_63<1; k_63+=(1)) {
                            (out + i * 1)[k_63] = (acc[k_63]);

                        }
                     
                          }

                        }
                    