
                          
                        #define C_CONTIGUOUS 1
#define USE_HALF 0

                        
                        extern "C" __global__ void GpuConv1DOnDevice(signed long int nx, signed long int ny, float *out, float **arg_3) {
    
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
                             for(int k_33=0; k_33<1; k_33+=(1)) {
                            acc[k_33] = (0.0f);

                        }
                     // acc = 0
                            
                            {
signed long int a=0;

for(signed long int v=0; v<1; v++) {
    xi[a] = arg_3[0][i*1+v];
     a++;
}
}
 // load xi variables from global memory to local thread memory
                          }

                          for (signed long int jstart = 0, tile = 0; jstart < ny; jstart += blockDim.x, tile++) {

                            // get the current column
                            signed long int j = tile * blockDim.x + threadIdx.x;

                            if (j < ny) { // we load yj from device global memory only if j<ny
                               
                            }
                            __syncthreads();

                            if (i < nx) { // we compute x1i only if needed
                              float * yjrel = yj;
                               for(int k_34=0; k_34<1; k_34+=(1)) {
                            tmp[k_34] = (0.0f);

                        }
                    
                              for (signed long int jrel = 0; (jrel < blockDim.x) && (jrel < ny - jstart); jrel++, yjrel += 0) {
                                
{
// Starting code block for IfElse(Var(0,1,0),Zero(1),Zero(1)).

float out_zero_0[1];

{
// Starting code block for Zero(1).

 for(int k_35=0; k_35<1; k_35+=(1)) {
                            out_zero_0[k_35] = (0.0f);

                        }
                    

// Finished code block for Zero(1).
}

float out_zero_1[1];

{
// Starting code block for Zero(1).

 for(int k_36=0; k_36<1; k_36+=(1)) {
                            out_zero_1[k_36] = (0.0f);

                        }
                    

// Finished code block for Zero(1).
}


 for(int k_37=0; k_37<1; k_37+=(1)) {
                            fout[(k_37*1)] = (
                    #ifdef __CUDACC__
                        (((xi+0)[(k_37*1)]>=0.0f) ? out_zero_0[(k_37*1)] : out_zero_1[(k_37*1)])
                    #else
                        (((xi+0)[(k_37*1)]>=0.0f) ? out_zero_0[(k_37*1)] : out_zero_1[(k_37*1)])
                    #endif
                );

                        }
                    

// Finished code block for IfElse(Var(0,1,0),Zero(1),Zero(1)).
}

 // Call the function, which outputs results in fout
                                
 for(int k_38=0; k_38<1; k_38+=(1)) {
                            tmp[(k_38*1)] += (fout[(k_38*1)]);

                        }
                    
                              }
                              
 for(int k_39=0; k_39<1; k_39+=(1)) {
                            acc[(k_39*1)] += (tmp[(k_39*1)]);

                        }
                    
                            }
                            __syncthreads();
                          }
                          if (i < nx) {
                            
 for(int k_40=0; k_40<1; k_40+=(1)) {
                            (out + i * 1)[k_40] = (acc[k_40]);

                        }
                     
                          }

                        }
                    