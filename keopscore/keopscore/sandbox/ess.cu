#define C_CONTIGUOUS 1
#define USE_HALF 0
extern "C" __global__ void GpuConv1DOnDevice(signed long int nx, signed long int ny, float* out, float** arg)
{
    signed long int i = (blockIdx.x*blockDim.x)+threadIdx.x;
    extern __shared__ float yj[];
    float fout[1];
    float xi[2];
    float acc[6];
    if((i<nx))
    {
        #pragma unroll(64)
        for(int k = 0; k<6; k += 2)
        {
            acc[k] = ( 1.0f/0.0f );
            acc[(k+1)] = 0.0f;
        }
        {
            signed long int a = 0;
            for(signed long int v = 0; v<2; v++)
            {
                xi[a] = (arg[0])[((i*2)+v)];
                a++;
            }
        }
    }
    for(signed long int jstart = 0,tile = 0; jstart<ny; jstart += blockDim.x,tile++)
        signed long int j = (tile*blockDim.x)+threadIdx.x;
        if((j<ny))
        {
            {
                signed long int a = 0;
                for(signed long int v = 0; v<2; v++)
                {
                    (yj + threadIdx.x * 2)[a] = (arg[1])[((j*2)+v)];
                    a++;
                }
            }
        }
        __syncthreads();
        if((i<nx))
        {
            float* yjrel = yj;
            for(signed long int jrel = 0; (jrel<blockDim.x)&&(jrel<(ny-jstart)); jrel++,yjrel += 2)
                {
                    // Starting code block for Sum((Var(0,2,0)-Var(1,2,1))**2);
                    float out_square[2];
                    {
                        // Starting code block for (Var(0,2,0)-Var(1,2,1))**2;
                        float out_subtract[2];
                        {
                            // Starting code block for Var(0,2,0)-Var(1,2,1);
                            #pragma unroll(64)
                            for(int k2 = 0; k2<2; k2++)
                                out_subtract[k2] = (xi+0)[k2]-(yjrel+0)[k2];
                                ;
                            // Finished code block for Var(0,2,0)-Var(1,2,1);
                        }
                        #pragma unroll(64)
                        for(int k3 = 0; k3<2; k3++)
                            out_square[k3] = out_subtract[k3]*out_subtract[k3];
                        // Finished code block for (Var(0,2,0)-Var(1,2,1))**2;
                    }
                    fout[0] = 0.0f;
                    #pragma unroll(64)
                    for(int k4 = 0; k4<2; k4++)
                        fout[0] += out_square[k4];
                    // Finished code block for Sum((Var(0,2,0)-Var(1,2,1))**2);
                }
                
                                    {
                                        float xik2;
                                        signed long int l2;
                                        #pragma unroll(64)
                                        for(signed long int k6=0; k6<1; k6++) {
                                            xik2 = fout[k6];
                                            #pragma unroll(64)                 
                                            for(l2=(k6+4); l2>=k6 && (xik2<acc[l2]); l2-=2) {
                                                float tmpl2 = acc[l2];
                                                signed long int indtmpl2 = acc[(l2+1)];
                                                acc[l2] = xik2;
                                                acc[(l2+1)] = jrel+(tile*blockDim.x);                      
                                                if(l2<(k6+4)) {
                                                    acc[(l2+2)] = tmpl2;
                                                    acc[((l2+2)+1)] = indtmpl2;
                                                }
                                            }
                                        }
                                    }
                                ;
        }
        __syncthreads();
    if((i<nx))
    {
        signed long int p = 0;
        #pragma unroll(64)
        for(int k7 = 0; k7<6; k7 += 2)
        {
            (out + i * 3)[p] = acc[(k7+1)];
            p++;
        }
    }
}