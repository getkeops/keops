#define C_CONTIGUOUS 1
#define USE_HALF 0
extern "C" __global__ void GpuConv1DOnDevice(signed long int nx, signed long int ny, float* out, float** arg2)
{
    signed long int i = ((blockIdx.x*blockDim.x)+threadIdx.x);
    extern __shared__ float yj[];
    float fout[3];
    float xi[4];
    float acc[3];
    float tmp[3];
    if((i<nx))
    {
        #pragma unroll(64)
        for(int k2 = 0; k2<3; k2++)
        {
            acc[k2] = 0.0f;
        }
        {
            signed long int a = 0;
            for(signed long int v = 0; v<3; v++)
            {
                xi[a] = (arg2[0])[((i*3)+v)];
                a++;
            }
            for(signed long int v = 0; v<1; v++)
            {
                xi[a] = (arg2[3])[(i+v)];
                a++;
            }
        }
    }
    for(signed long int jstart = 0,tile = 0; jstart<ny; jstart += blockDim.x,tile++)
    {
        signed long int j = (tile*blockDim.x)+threadIdx.x;
        if((j<ny))
        {
            {
                signed long int a = 0;
                for(signed long int v = 0; v<3; v++)
                {
                    (yj + threadIdx.x * 4)[a] = (arg2[1])[((j*3)+v)];
                    a++;
                }
                for(signed long int v = 0; v<1; v++)
                {
                    (yj + threadIdx.x * 4)[a] = (arg2[2])[(j+v)];
                    a++;
                }
            }
        }
        __syncthreads();
        if((i<nx))
        {
            float* yjrel = yj;
            #pragma unroll(64)
            for(int k3 = 0; k3<3; k3++)
            {
                tmp[k3] = 0.0f;
            }
            for(signed long int jrel = 0; (jrel<blockDim.x)&&(jrel<(ny-jstart)); jrel++,yjrel += 4)
            {
                // Starting code block for (-2*((Var(0,3,0)-Var(1,3,1))*(Var(3,1,0)*Var(2,1,1))))*Exp(-Sum((Var(0,3,0)-Var(1,3,1))**2));
                float out_sum2;
                // Starting code block for Sum((Var(0,3,0)-Var(1,3,1))**2);
                out_sum2 = 0.0f;
                #pragma unroll(64)
                for(int k5 = 0; k5<3; k5++)
                {
                    float out_square2;
                    out_square2 = (xi+0)[k5]-(yjrel+0)[k5];
                    out_square2 *= out_square2;
                    out_sum2 += out_square2;
                }
                // Finished code block for Sum((Var(0,3,0)-Var(1,3,1))**2);
                #pragma unroll(64)
                for(int k4 = 0; k4<3; k4++)
                {
                    fout[k4] = (-2*(((xi+0)[k4]-(yjrel+0)[k4])*((xi+3)[0]*(yjrel+3)[0])))*(exp((-out_sum2)));
                }
                // Finished code block for (-2*((Var(0,3,0)-Var(1,3,1))*(Var(3,1,0)*Var(2,1,1))))*Exp(-Sum((Var(0,3,0)-Var(1,3,1))**2));
                #pragma unroll(64)
                for(int k6 = 0; k6<3; k6++)
                {
                    tmp[k6] += fout[k6];
                }
            }
            #pragma unroll(64)
            for(int k7 = 0; k7<3; k7++)
            {
                acc[k7] += tmp[k7];
            }
        }
        __syncthreads();
    }
    if((i<nx))
    {
        #pragma unroll(64)
        for(int k8 = 0; k8<3; k8++)
        {
            (out + i * 3)[k8] = acc[k8];
        }
    }
}