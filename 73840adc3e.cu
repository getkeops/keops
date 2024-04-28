#define C_CONTIGUOUS 1
#define USE_HALF 0
extern "C" __global__ void GpuConv1DOnDevice(signed long int nx, signed long int ny, float* out, float** arg)
{
    signed long int i = (blockIdx.x*blockDim.x)+threadIdx.x;
    extern __shared__ float yj[];
    float fout;
    float xi[3];
    float acc;
    float tmp;
    if(i<nx)
    {
        acc = 0.0f;
        signed long int a = 0;
        {
            for(signed long int v = 0; v<3; v++)
            {
                xi[a] = arg[0][(i*3)+v];
                a++;
            }
        }
    }
    for(signed long int jstart = 0,tile = 0; jstart<ny; jstart += blockDim.x,tile++)
    {
        signed long int j = (tile*blockDim.x)+threadIdx.x;
        if(j<ny)
        {
            signed long int a = 0;
            {
                for(signed long int v = 0; v<3; v++)
                {
                    (yj+(threadIdx.x*4))[a] = arg[1][(j*3)+v];
                    a++;
                }
                for(signed long int v = 0; v<1; v++)
                {
                    (yj+(threadIdx.x*4))[a] = arg[2][j+v];
                    a++;
                }
            }
        }
        __syncthreads();
        if(i<nx)
        {
            float* yjrel = yj;
            tmp = 0.0f;
            for(signed long int jrel = 0; (jrel<blockDim.x)&&(jrel<(ny-jstart)); jrel++,yjrel += 4)
            {
                // Starting code block for Var(2,1,1)*Exp(-Sum(Pow(Var(0,3,0)-Var(1,3,1))));
                float out_sum;
                // Starting code block for Sum(Pow(Var(0,3,0)-Var(1,3,1)));
                out_sum = 0.0f;
                #pragma unroll(64)
                for(int k = 0; k<3; k++)    out_sum += 
                                            #ifdef __CUDACC__
                                                powf((xi[k]-yjrel[k]),(3))
                                            #else
                                                pow((xi[k]-yjrel[k]),(3))
                                            #endif
                                        ;
                // Finished code block for Sum(Pow(Var(0,3,0)-Var(1,3,1)));
                fout = (yjrel+3)[0]*(exp((-out_sum)));
                // Finished code block for Var(2,1,1)*Exp(-Sum(Pow(Var(0,3,0)-Var(1,3,1))));
                tmp += fout;
            }
            acc += tmp;
        }
        __syncthreads();
    }
    if(i<nx)    (out+i)[0] = acc;
}