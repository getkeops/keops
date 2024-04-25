#define C_CONTIGUOUS 1
#define USE_HALF 0
extern "C" __global__ void GpuConv1DOnDevice(signed long int nx, signed long int ny, float* out, float** arg)
{
    signed long int i = (signed long int)((blockIdx.x*blockDim.x)+threadIdx.x);
    extern __shared__ float yj[];
    float fout[1];
    float acc[1];
    float tmp[1];
    if((i<nx))
    {
        acc[0] = 0.0f;
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
                    (yj + threadIdx.x * 1)[a] = (arg[0])[(j+v)];
                    a++;
                }
            }
        }
        __syncthreads();
        if((i<nx))
        {
            float* yjrel = yj;
            tmp[0] = 0.0f;
            for(signed long int jrel = (signed long int)0; (jrel<blockDim.x)&&(jrel<(ny-jstart)); jrel++,yjrel++)
            {
                // Starting code block for IfElse(Var(0,1,0),Sqrt(Var(0,1,0)),Var(0,1,0)**2*Var(0,1,0));
                float out_square;
                out_square = (yjrel+0)[0];
                out_square *= out_square;
                fout[0] = (((yjrel+0)[0]>=0.0f) ? (sqrt((yjrel+0)[0])) : (out_square*(yjrel+0)[0]));
                // Finished code block for IfElse(Var(0,1,0),Sqrt(Var(0,1,0)),Var(0,1,0)**2*Var(0,1,0));
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