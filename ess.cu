#define C_CONTIGUOUS 1
#define USE_HALF 1
#define half2 __half2
#include <cuda_fp16.h>
extern "C" __global__ void GpuConv1DOnDevice(signed long int nx, signed long int ny, half2* out, half2** arg)
{
    signed long int i = (signed long int)((blockIdx.x*blockDim.x)+threadIdx.x);
    extern __shared__ half2 yj[];
    half2 fout[1];
    half2 xi[3];
    half2 acc[1];
    half2 tmp[1];
    if((i<nx))
    {
        acc[0] = __float2half2_rn(0.0f);
        {
            signed long int a = (signed long int)0;
            for(signed long int v = (signed long int)0; v<3; v++)
            {
                xi[a] = (arg[1])[((i*3)+v)];
                a++;
            }
        }
    }
    for(signed long int jstart = (signed long int)0,tile = (signed long int)0; jstart<ny; jstart += (signed long int)blockDim.x,tile++)
    {
        signed long int j = (tile*blockDim.x)+threadIdx.x;
        if((j<ny))
        {
            {
                signed long int a = (signed long int)0;
                for(signed long int v = (signed long int)0; v<3; v++)
                {
                    (yj + threadIdx.x * 3)[a] = (arg[0])[((j*3)+v)];
                    a++;
                }
            }
        }
        __syncthreads();
        if((i<nx))
        {
            half2* yjrel = yj;
            tmp[0] = __float2half2_rn(0.0f);
            for(signed long int jrel = (signed long int)0; (jrel<blockDim.x)&&(jrel<(ny-jstart)); jrel++,yjrel += 3)
            {
                // Starting code block for Exp(-Sum(Var(0,3,0)-Var(1,3,1)));
                half2 out_sum;
                // Starting code block for Sum(Var(0,3,0)-Var(1,3,1));
                out_sum = __float2half2_rn(0.0f);
                #pragma unroll(64)
                for(int k = 0; k<3; k++)
                {
                    out_sum += 
                                            #ifdef __CUDACC__
                                                __hsub2((yjrel+0)[k],(xi+0)[k])
                                            #else
                                                (yjrel+0)[k]-(xi+0)[k]
                                            #endif
                                        ;
                }
                // Finished code block for Sum(Var(0,3,0)-Var(1,3,1));
                fout[0] = 
                                        #ifdef __CUDACC__
                                            h2exp((
                                        #ifdef __CUDACC__
                                            __hneg2(out_sum)
                                        #else
                                            -out_sum
                                        #endif
                                    ))
                                        #else
                                            exp((
                                        #ifdef __CUDACC__
                                            __hneg2(out_sum)
                                        #else
                                            -out_sum
                                        #endif
                                    ))
                                        #endif
                                    ;
                // Finished code block for Exp(-Sum(Var(0,3,0)-Var(1,3,1)));
                tmp[0] += fout[0];
            }
            acc[0] += tmp[0];
        }
        __syncthreads();
    }
    if((i<nx))
    {
        (out + i * 1)[0] = acc[0];
    }
}