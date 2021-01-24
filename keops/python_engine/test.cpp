#include <cmath>

#ifdef USE_OPENMP
#include <omp.h>
#endif

extern "C" int CpuReduc(int nx, int ny, float* out , float* arg0, float* arg1, float* arg2, float* arg3, float* arg4, float* arg5) {
    float* args[6];
    args[0] = arg0;
args[1] = arg1;
args[2] = arg2;
args[3] = arg3;
args[4] = arg4;
args[5] = arg5;

    
    
    #pragma omp parallel for
    for (int i = 0; i < nx; i++) {
        float fout[1];
        float xi[2];
        float yj[2];
        float acc[1];
        xi[0] = args[0][i*2+0];
xi[1] = args[0][i*2+1];

        #pragma unroll
for(int k=0; k<1; k++)
    acc[k] = (float)(0.0f);
        for (int j = 0; j < ny; j++) {
            yj[0] = args[1][j*2+0];
yj[1] = args[1][j*2+1];

            
{
// Starting code block for Exp(-Sum((Var(0,2,0)-Var(1,2,1))**2)).

float out_minus_1[1];

{
// Starting code block for -Sum((Var(0,2,0)-Var(1,2,1))**2).

float out_sum_1[1];

{
// Starting code block for Sum((Var(0,2,0)-Var(1,2,1))**2).

float out_square_1[2];

{
// Starting code block for (Var(0,2,0)-Var(1,2,1))**2.

float out_subtract_1[2];

{
// Starting code block for Var(0,2,0)-Var(1,2,1).

#pragma unroll
for(int k=0; k<2; k++) {
    out_subtract_1[k*1] = (xi+0)[k*1]-(yj+0)[k*1];
 }


// Finished code block for Var(0,2,0)-Var(1,2,1).
}

#pragma unroll
for(int k=0; k<2; k++) {
    out_square_1[k*1] = out_subtract_1[k*1]*out_subtract_1[k*1];
 }


// Finished code block for (Var(0,2,0)-Var(1,2,1))**2.
}

*out_sum_1 = (float)(0.0f);
#pragma unroll
for(int k=0; k<2; k++) {
    out_sum_1[k*0] += out_square_1[k*1];
 }


// Finished code block for Sum((Var(0,2,0)-Var(1,2,1))**2).
}

#pragma unroll
for(int k=0; k<1; k++) {
    out_minus_1[k*1] = -out_sum_1[k*1];
 }


// Finished code block for -Sum((Var(0,2,0)-Var(1,2,1))**2).
}

#pragma unroll
for(int k=0; k<1; k++) {
    fout[k*1] = exp(out_minus_1[k*1]);
 }


// Finished code block for Exp(-Sum((Var(0,2,0)-Var(1,2,1))**2)).
}


            #pragma unroll
for(int k=0; k<1; k++) {
    acc[k*1] += (float)(fout[k*1]); }

        }
        #pragma unroll
for(int k=0; k<1; k++)
    (out + i * 1)[k] = (float)acc[k];

    }
    return 0;
}

