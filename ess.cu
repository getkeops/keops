#define C_CONTIGUOUS 1
#define USE_HALF 0
extern "C" __global__ void GpuConv1DOnDevice(signed long int nx, signed long int ny, float* out, float** arg_1)
{
    // get the index of the current thread
    signed long int i = ((blockIdx.x*blockDim.x)+threadIdx.x);
    // declare shared mem
    extern __shared__ float yj[];
    float fout[3];
    float xi[4];
    float acc[3];
    float tmp[3];
    if((i<nx))
    {
        #pragma unroll(64)
        for(int var_12 = 0; (var_12<3); var_12 += 1)
        {
            acc[var_12] = 0.0f;
        }

        {
            signed long int a = 0;
            for(signed long int v = 0; (v<3); v++)
            {
                xi[a] = ((arg_1[0])[((i*3)+v)]);
                a++;
            }
            for(signed long int v = 0; (v<1); v++)
            {
                xi[a] = ((arg_1[3])[((i*1)+v)]);
                a++;
            }
        }
    }
    for(signed long int jstart = 0,tile = 0; (jstart<ny); jstart += blockDim.x,tile++)
    {
        // get the current column
        signed long int j = ((tile*blockDim.x)+threadIdx.x);

        // we load yj from device global memory only if j<ny
        if((j<ny))
        {

            {
                signed long int a = 0;
                for(signed long int v = 0; (v<3); v++)
                {
                    (yj + threadIdx.x * 4)[a] = ((arg_1[1])[((j*3)+v)]);
                    a++;
                }
                for(signed long int v = 0; (v<1); v++)
                {
                    (yj + threadIdx.x * 4)[a] = ((arg_1[2])[((j*1)+v)]);
                    a++;
                }
            }
        }
        __syncthreads();

        // we compute x1i only if needed
        if((i<nx))
        {
            float* yjrel = yj;
            #pragma unroll(64)
            for(int var_13 = 0; (var_13<3); var_13 += 1)
            {
                tmp[var_13] = 0.0f;
            }
            for(signed long int jrel = 0; ((jrel<blockDim.x)&&(jrel<(ny-jstart))); jrel++,yjrel += 4)
            {

                {
                    // Starting code block for (-2*((Var(0,3,0)-Var(1,3,1))*(Var(3,1,0)*Var(2,1,1))))*Exp(-Sum((Var(0,3,0)-Var(1,3,1))**2));
                    float out_mult_0[3];

                    {
                        // Starting code block for -2*((Var(0,3,0)-Var(1,3,1))*(Var(3,1,0)*Var(2,1,1)));
                        float out_intcst_0[1];

                        {
                            // Starting code block for -2;
                            (*out_intcst_0) = (float)(float)-2;
                            // Finished code block for -2;
                        }
                        float out_mult_1[3];

                        {
                            // Starting code block for (Var(0,3,0)-Var(1,3,1))*(Var(3,1,0)*Var(2,1,1));
                            float out_subtract_1[3];

                            {
                                // Starting code block for Var(0,3,0)-Var(1,3,1);
                                #pragma unroll(64)
                                for(int var_14 = 0; (var_14<3); var_14 += 1)
                                {
                                    out_subtract_1[(var_14*1)] = (xi+0)[(var_14*1)]-(yjrel+0)[(var_14*1)];
                                    ;
                                }
                                // Finished code block for Var(0,3,0)-Var(1,3,1);
                            }
                            float out_mult_2[1];

                            {
                                // Starting code block for Var(3,1,0)*Var(2,1,1);
                                #pragma unroll(64)
                                for(int var_15 = 0; (var_15<1); var_15 += 1)
                                {
                                    out_mult_2[(var_15*1)] = 
                                                        #ifdef __CUDACC__
                                                            ((xi+3)[(var_15*1)]*(yjrel+3)[(var_15*1)])
                                                        #else
                                                            ((xi+3)[(var_15*1)]*(yjrel+3)[(var_15*1)])
                                                        #endif
                                                    ;
                                }
                                // Finished code block for Var(3,1,0)*Var(2,1,1);
                            }
                            #pragma unroll(64)
                            for(int var_16 = 0; (var_16<3); var_16 += 1)
                            {
                                out_mult_1[(var_16*1)] = 
                                                    #ifdef __CUDACC__
                                                        (out_subtract_1[(var_16*1)]*out_mult_2[(var_16*0)])
                                                    #else
                                                        (out_subtract_1[(var_16*1)]*out_mult_2[(var_16*0)])
                                                    #endif
                                                ;
                            }
                            // Finished code block for (Var(0,3,0)-Var(1,3,1))*(Var(3,1,0)*Var(2,1,1));
                        }
                        #pragma unroll(64)
                        for(int var_17 = 0; (var_17<3); var_17 += 1)
                        {
                            out_mult_0[(var_17*1)] = 
                                                #ifdef __CUDACC__
                                                    (out_intcst_0[(var_17*0)]*out_mult_1[(var_17*1)])
                                                #else
                                                    (out_intcst_0[(var_17*0)]*out_mult_1[(var_17*1)])
                                                #endif
                                            ;
                        }
                        // Finished code block for -2*((Var(0,3,0)-Var(1,3,1))*(Var(3,1,0)*Var(2,1,1)));
                    }
                    float out_exp_1[1];

                    {
                        // Starting code block for Exp(-Sum((Var(0,3,0)-Var(1,3,1))**2));
                        float out_minus_1[1];

                        {
                            // Starting code block for -Sum((Var(0,3,0)-Var(1,3,1))**2);
                            float out_sum_1[1];

                            {
                                // Starting code block for Sum((Var(0,3,0)-Var(1,3,1))**2);
                                float out_square_1[3];

                                {
                                    // Starting code block for (Var(0,3,0)-Var(1,3,1))**2;
                                    float out_subtract_2[3];

                                    {
                                        // Starting code block for Var(0,3,0)-Var(1,3,1);
                                        #pragma unroll(64)
                                        for(int var_18 = 0; (var_18<3); var_18 += 1)
                                        {
                                            out_subtract_2[(var_18*1)] = (xi+0)[(var_18*1)]-(yjrel+0)[(var_18*1)];
                                            ;
                                        }
                                        // Finished code block for Var(0,3,0)-Var(1,3,1);
                                    }
                                    #pragma unroll(64)
                                    for(int var_19 = 0; (var_19<3); var_19 += 1)
                                    {
                                        out_square_1[(var_19*1)] = (out_subtract_2[(var_19*1)]*out_subtract_2[(var_19*1)]);
                                    }
                                    // Finished code block for (Var(0,3,0)-Var(1,3,1))**2;
                                }
                                #pragma unroll(64)
                                for(int var_20 = 0; (var_20<1); var_20 += 1)
                                {
                                    out_sum_1[var_20] = 0.0f;
                                }
                                #pragma unroll(64)
                                for(int var_21 = 0; (var_21<3); var_21 += 1)
                                {
                                    out_sum_1[(var_21*0)] += out_square_1[(var_21*1)];
                                }
                                // Finished code block for Sum((Var(0,3,0)-Var(1,3,1))**2);
                            }
                            #pragma unroll(64)
                            for(int var_22 = 0; (var_22<1); var_22 += 1)
                            {
                                out_minus_1[(var_22*1)] = -out_sum_1[(var_22*1)];
                                ;
                            }
                            // Finished code block for -Sum((Var(0,3,0)-Var(1,3,1))**2);
                        }
                        #pragma unroll(64)
                        for(int var_23 = 0; (var_23<1); var_23 += 1)
                        {
                            out_exp_1[(var_23*1)] = 
                                                #ifdef __CUDACC__
                                                    exp(out_minus_1[(var_23*1)])
                                                #else
                                                    exp(out_minus_1[(var_23*1)])
                                                #endif
                                            ;
                        }
                        // Finished code block for Exp(-Sum((Var(0,3,0)-Var(1,3,1))**2));
                    }
                    #pragma unroll(64)
                    for(int var_24 = 0; (var_24<3); var_24 += 1)
                    {
                        fout[(var_24*1)] = 
                                            #ifdef __CUDACC__
                                                (out_mult_0[(var_24*1)]*out_exp_1[(var_24*0)])
                                            #else
                                                (out_mult_0[(var_24*1)]*out_exp_1[(var_24*0)])
                                            #endif
                                        ;
                    }
                    // Finished code block for (-2*((Var(0,3,0)-Var(1,3,1))*(Var(3,1,0)*Var(2,1,1))))*Exp(-Sum((Var(0,3,0)-Var(1,3,1))**2));
                }
                #pragma unroll(64)
                for(int var_25 = 0; (var_25<3); var_25 += 1)
                {
                    tmp[(var_25*1)] += fout[(var_25*1)];
                }
            }
            #pragma unroll(64)
            for(int var_26 = 0; (var_26<3); var_26 += 1)
            {
                acc[(var_26*1)] += tmp[(var_26*1)];
            }
        }
        __syncthreads();
    }
    if((i<nx))
    {
        #pragma unroll(64)
        for(int var_27 = 0; (var_27<3); var_27 += 1)
        {
            (out + i * 3)[var_27] = acc[var_27];
        }
    }
}