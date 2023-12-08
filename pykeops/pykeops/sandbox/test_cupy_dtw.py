import cupy as cp
import numpy as np
from time import time
from cupyx.profiler import benchmark

cp.cuda.runtime.deviceSynchronize()

loaded_from_source = r'''

#define MIN(x,y) (((x) < (y)) ? (x) : (y))
#define MIN(x,y,z) (MIN(x,MIN(y,z)))

#include <cooperative_groups.h>
using namespace cooperative_groups;

extern "C"{

__global__ void dtw(int nx, int ny, float *x, float *y, float *out) 
{   
    int blocksize = blockDim.x;
    int iblock = blockIdx.x;

    // center buffer
    float *bufref = out + nx - iblock * blocksize;
    float *buf = bufref;

    // get the index of the current thread
    int iloc = threadIdx.x;
    int i = indblock * blocksize + iloc;
    int ibuf = -iloc-1;
    
    // declare shared mem - size to allocate must be 4*blocksize+1
    extern __shared__ float shared[];
    float *yloc = shared;
    float *bufloc = yloc + 2*blocksize + 1;
    int buflocsize = 3*blocksize + 1;

    float xi, d2ij;

    if (i < nx) {
        xi = x[i];
        bufloc[ibuf] = buf[ibuf];
        if (iloc==0)
            bufloc[0] = buf[0];

        for (int jblock=0, jstart=0; jstart<ny; jblock++, jstart+=blocksize)
        {
            // recenter buffer
            buf = bufref + jblock * blocksize;

            // get the current column
            int jloc = iloc;
            int j = jblock * blocksize + jloc;
            int jbuf = jloc+1;
            int jbufloc = jbuf+(jblock%2)*blocksize;

            if (j < ny) 
            {
                yloc[jloc] = y[j];
                bufloc[jbufloc] = buf[jbuf];
            }
            __syncthreads();

            for ( int jloc = 0, j=jstart; (jloc < blocksize) && (j < ny); jloc++, j++) 
            {
                d2ij = yloc[jloc]-xi;
                d2ij *= d2ij;
                int k = jloc - iloc + (jblock%2)*blocksize;
                int km1 = (k-1)%buflocsize;
                int kp1 = (k+1)%buflocsize;
                bufloc[k] = d2ij + min(bufloc[km1],bufloc[k],bufloc[kp1]);
            }
    }
    __syncthreads();

    if (i < nx) 
    {
      buf[ibuf] = bufloc[ibuf];
      if (iloc==0)
        buf[0] = bufloc[0];
    }

    if (j < ny) 
    {
        buf[jbuf] = bufloc[jbuf];
    }

  }

}'''

ker_dtw = cp.RawKernel(loaded_from_source, "dtw")

def ker_dtw_raw(x,y):
    N = len(x)
    buf = cp.zeros(2*N+1, dtype=cp.float32)
    buf[:] = cp.infty
    blocksize = 192
    gridsize = 1+(N-1)//blocksize
    shared_mem = 3*blocksize+1
    ker_dtw((gridsize,gridsize),(blocksize,blocksize),(N,N,x,y,buf),shared_mem=shared_mem)
    cp.cuda.runtime.deviceSynchronize()
    return buf

#def ker_dtw_cupy(out,x,y):

def bench_time(fun,args,n_repeat=1):
    for k in range(2):
        fun(*args) # warmup
    start  = time()
    fun(*args)
    end  = time()
    return f"time for {fun.__name__} : {end-start}"

N = 1000
n_repeat = 1
x = cp.random.rand(N, dtype=cp.float32)
y = cp.random.rand(N, dtype=cp.float32)

for bench_method in (bench_time,benchmark):
    print(f"\n----------------------\nUsing {bench_method.__name__}\n---------------------")
    print(bench_method(ker_dtw_raw,(x,y),n_repeat=n_repeat))
    #if N<20000:
    #    print(bench_method(ker_dtw_cupy,(out_ref,x,y,b),n_repeat=n_repeat))
    #    print("relative error : ", cp.linalg.norm(out-out_ref)/cp.linalg.norm(out_ref))