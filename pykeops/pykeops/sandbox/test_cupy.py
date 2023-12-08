import cupy as cp
import numpy as np
from time import time
from cupyx.profiler import benchmark
import torch
from pykeops.torch import Genred

cp.cuda.runtime.deviceSynchronize()

loaded_from_source = r'''
extern "C"{

__global__ void gauss_conv(int nx, int ny, float *out, float *x, float *y, float *b) 
{   
    // get the index of the current thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // declare shared mem
    extern __shared__ float yj[];

    float fout;
    float xi;
    float acc;

    if (i < nx) {
      acc = 0;
      xi = x[i];
    }

    for ( int jstart = 0, tile = 0; jstart < ny; jstart += blockDim.x, tile++) 
    {

        // get the current column
        int j = tile * blockDim.x + threadIdx.x;

        if (j < ny) 
        { // we load yj from device global memory only if j<ny
            yj[threadIdx.x*2] = y[j];
            yj[threadIdx.x*2+1] = b[j];
        }
        __syncthreads();

        if (i < nx) {
            float *yjrel = yj;
            for ( int jrel = 0; (jrel < blockDim.x) && (jrel < ny - jstart); jrel++, yjrel += 2) 
            {
                float d = yjrel[0]-xi;
                d = exp(-d*d)*yjrel[1];
                acc += d;
            }
        }
        __syncthreads();
    }
    if (i < nx) 
    {
      out[i] =  acc;
    }

  }

}'''
ker_gauss = cp.RawKernel(loaded_from_source, "gauss_conv")

def ker_gauss_raw(out,x,y,b):
    N = len(x)
    blocksize = 192
    gridsize = 1+(N-1)//blocksize
    shared_mem = 4*blocksize*2
    ker_gauss((gridsize,),(blocksize,),(N,N,out,x,y,b),shared_mem=shared_mem)
    cp.cuda.runtime.deviceSynchronize()

def ker_gauss_cupy(out,x,y,b):
    cp.sum(cp.exp(-(x[:,None]-y[None,:])**2) * b[None,:],axis=1,out=out)
    cp.cuda.runtime.deviceSynchronize()

def ker_gauss_torch(out,x,y,b):
    torch.sum(torch.exp(-(x[:,None]-y[None,:])**2) * b[None,:],axis=1,out=out)
    cp.cuda.runtime.deviceSynchronize()

fun_keops = Genred("Exp(-Square(X-Y))*B", ["X=Vi(0,1)", "Y=Vj(1,1)", "B=Vj(2,1)"], axis=1)
def ker_gauss_keops(out,x,y,b): 
    fun_keops(x[:,None],y[:,None],b[:,None],out=out)
    cp.cuda.runtime.deviceSynchronize()

def bench_time(fun,args,n_repeat=1):
    for k in range(2):
        fun(*args) # warmup
    start  = time()
    fun(*args)
    end  = time()
    return f"time for {fun.__name__} : {end-start}"

N = 100000
n_repeat = 10
x = cp.random.rand(N, dtype=cp.float32)
y = cp.random.rand(N, dtype=cp.float32)
b = cp.random.rand(N, dtype=cp.float32)
out = cp.zeros(N, dtype=cp.float32)
out_ref = cp.zeros(N, dtype=cp.float32)

xt = torch.as_tensor(x, device='cuda')
yt = torch.as_tensor(y, device='cuda')
bt = torch.as_tensor(b, device='cuda')
outt = torch.as_tensor(out, device='cuda')

for bench_method in (bench_time,benchmark):
    print(f"\n----------------------\nUsing {bench_method.__name__}\n---------------------")
    print(bench_method(ker_gauss_raw,(out_ref,x,y,b),n_repeat=n_repeat))
    print(bench_method(ker_gauss_keops,(outt,xt,yt,bt),n_repeat=n_repeat))
    print("relative error : ", cp.linalg.norm(outt-out_ref)/cp.linalg.norm(out_ref))
    if N<20000:
        print(bench_method(ker_gauss_cupy,(out,x,y,b),n_repeat=n_repeat))
        print(bench_method(ker_gauss_torch,(outt,xt,yt,bt),n_repeat=n_repeat))