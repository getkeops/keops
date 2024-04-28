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
                    (yj+(threadIdx.x*7))[a] = arg[1][(j*3)+v];
                    a++;
                }
                for(signed long int v = 0; v<3; v++)
                {
                    (yj+(threadIdx.x*7))[a] = arg[2][(j*3)+v];
                    a++;
                }
                for(signed long int v = 0; v<1; v++)
                {
                    (yj+(threadIdx.x*7))[a] = arg[3][j+v];
                    a++;
                }
            }
        }
        __syncthreads();
        if(i<nx)
        {
            float* yjrel = yj;
            tmp = 0.0f;
            for(signed long int jrel = 0; (jrel<blockDim.x)&&(jrel<(ny-jstart)); jrel++,yjrel += 7)
            {
                // Starting code block for Var(3,1,1)*Exp(-(Atan2(Var(0,3,0),Var(1,3,1))|(Var(0,3,0)*(Var(1,3,1)*Var(2,3,1)))**2));
                float out_scalprod;
                // Starting code block for Atan2(Var(0,3,0),Var(1,3,1))|(Var(0,3,0)*(Var(1,3,1)*Var(2,3,1)))**2;
                out_scalprod = 0.0f;
                #pragma unroll(64)
                for(int k = 0; k<3; k++)
                {
                    float out_square;
                    out_square = xi[k]*(yjrel[k]*(yjrel+3)[k]);
                    out_square *= out_square;
                    out_scalprod = 
                                            #ifdef __CUDACC__
                                                fmaf((atan2(xi[k],yjrel[k])),out_square,out_scalprod)
                                            #else
                                                fma((atan2(xi[k],yjrel[k])),out_square,out_scalprod)
                                            #endif
                                        ;
                }
                // Finished code block for Atan2(Var(0,3,0),Var(1,3,1))|(Var(0,3,0)*(Var(1,3,1)*Var(2,3,1)))**2;
                fout = (yjrel+6)[0]*(exp((-out_scalprod)));
                // Finished code block for Var(3,1,1)*Exp(-(Atan2(Var(0,3,0),Var(1,3,1))|(Var(0,3,0)*(Var(1,3,1)*Var(2,3,1)))**2));
                tmp += fout;
            }
            acc += tmp;
        }
        __syncthreads();
    }
    if(i<nx)    (out+i)[0] = acc;
}