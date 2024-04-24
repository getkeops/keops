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
                xi[a] = (arg[0])[((i*3)+v)];
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
                for(signed long int v = (signed long int)0; v<1; v++)
                {
                    (yj + threadIdx.x * 1)[a] = (arg[1])[(j+v)];
                    a++;
                }
            }
        }
        __syncthreads();
        if((i<nx))
        {
            half2* yjrel = yj;
            tmp[0] = __float2half2_rn(0.0f);
            for(signed long int jrel = (signed long int)0; (jrel<blockDim.x)&&(jrel<(ny-jstart)); jrel++,yjrel++)
            {
                // Starting code block for Sum((Var(0,3,0)<Var(1,1,1))!=(1<=Var(1,1,1)));
                fout[0] = __float2half2_rn(0.0f);
                #pragma unroll(64)
                for(int k = 0; k<3; k++)
                {
                    fout[0] += 
                                            #ifdef __CUDACC__
                                                __hne2((
                                            #ifdef __CUDACC__
                                                __hlt2((xi+0)[k],(yjrel+0)[0])
                                            #else
                                                (((xi+0)[k]<(yjrel+0)[0])? 1.0f : 0.0f)
                                            #endif
                                        ),(
                                            #ifdef __CUDACC__
                                                __hle2(__float2half2_rn(1),(yjrel+0)[0])
                                            #else
                                                ((__float2half2_rn(1)<=(yjrel+0)[0])? 1.0f : 0.0f)
                                            #endif
                                        ))
                                            #else
                                                (((
                                            #ifdef __CUDACC__
                                                __hlt2((xi+0)[k],(yjrel+0)[0])
                                            #else
                                                (((xi+0)[k]<(yjrel+0)[0])? 1.0f : 0.0f)
                                            #endif
                                        )!=(
                                            #ifdef __CUDACC__
                                                __hle2(__float2half2_rn(1),(yjrel+0)[0])
                                            #else
                                                ((__float2half2_rn(1)<=(yjrel+0)[0])? 1.0f : 0.0f)
                                            #endif
                                        ))? 1.0f : 0.0f)
                                            #endif
                                        ;
                }
                // Finished code block for Sum((Var(0,3,0)<Var(1,1,1))!=(1<=Var(1,1,1)));
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