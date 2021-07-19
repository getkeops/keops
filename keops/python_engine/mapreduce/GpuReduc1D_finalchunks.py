from keops.python_engine.mapreduce.MapReduce import MapReduce
from keops.python_engine.mapreduce.GpuAssignZero import GpuAssignZero
from keops.python_engine.utils.code_gen_utils import (
    c_variable,
    c_array,
    c_include,
    signature_list,
    call_list,
    load_vars,
    load_vars_chunks,
    sizeof,
    pointer,
    table,
    table4
)
from keops.python_engine.formulas.reductions.sum_schemes import *
from keops.python_engine.compilation import Gpu_link_compile
from keops.python_engine import cuda_block_size, dimchunk
from .Chunk_Mode_Constants import Chunk_Mode_Constants

        
def do_finalchunk_sub(dtype, varfinal, dimfinalchunk_curr,
                                acc, i, j, jstart, chunk, nx, ny, arg, fout, yj, out):
    
    dimout = varfinal.dim
    yjloc = c_variable(pointer(dtype), f"({yj.id} + threadIdx.x * {dimfinalchunk})")
    load_chunks_routine_j = load_vars_chunks([varfinal.n], dimfinalchunk, dimfinalchunk_curr, varfinal.dim,
                                                yjloc, arg, chunk, row_index=j)    
    return f"""
                {acc.assign(c_zero_float)}
                {dtype} *yjrel = yj;
                if ({j.id} < {ny.id}) {{ // we load yj from device global memory only if j<ny
                    {load_chunks_routine_j}
                }}
                __syncthreads();
                for (int jrel = 0; (jrel < blockDim.x) && (jrel < {ny.id} - jstart); jrel++, yjrel += {dimfinalchunk}) {{          
                    if ({i.id} < {nx.id}) {{ // we compute only if needed
                        #pragma unroll
                        for (int k=0; k<{dimfinalchunk_curr}; k++)
                            {acc.id}[k] += yjrel[k] * fout[jrel];
                    }}
                    __syncthreads();
                }}
                if ({i.id} < {nx.id}) {{
                    #pragma unroll
                    for (int k=0; k<{{dimfinalchunk_curr}}; k++)
                        {out.id}[i*{dimout}+{chunk.id}*{dimfinalchunk}+k] += {acc.id}[k];
                }}
                __syncthreads();
            """
    

class GpuReduc1D_finalchunks(MapReduce, Gpu_link_compile):
    # class for generating the final C++ code, Gpu version

    AssignZero = GpuAssignZero

    def __init__(self, *args):
        MapReduce.__init__(self, *args)
        Gpu_link_compile.__init__(self)
        self.blocksize_chunks = min(cuda_block_size, 1024, 49152 // max(1, self.dimy*sizeof(self.dtype)))
        
        

    def get_code(self):

        super().get_code()
        formula = self.red_formula.formula
        blocksize_chunks = self.blocksize_chunks
        varfinal = None
        nchunks = 1 + (varfinal.dim-1) // dimfinalchunk
        dimlastfinalchunk = varfinal.dim - (nchunks-1)*dimfinalchunk
        dimsx = self.dimsx
        dimsy = self.dimsy
        dimsp = self.dimsp
        indsi = self.indsi
        indsj = self.indsj
        indsp = self.indsp
        dimx = sum(dimsx)
        dimy = sum(dimsy)
        dimp = sum(dimsp)
        dimout = varfinal.dim
        dimfout = self.dimfout
        if dimfout != 1:
            raise ValueError("dimfout should be 1")
        sum_scheme  self.sum_scheme
        if sum_scheme != block_sum:
            raise ValueError("only block_sum available")
        param_loc = c_array(dtype, dimp, "param_loc")
        fout = c_array(dtype, dimfout*blocksize_chunks, "fout")
        xi = c_array(dtype, dimx, "xi")
        acc = c_array(dtypeacc, dimfinalchunk, "acc")
        yjloc = c_array(dtype, dimy, f"(yj + threadIdx.x * {dimy})")
        foutjrel = c_array(dtype, dimfout, f"({fout.id}+jrel*{dimfout})")
        yjrel = c_array(dtype, dimy, "yjrel")
        table = self.varloader.table(xi, yjrel, param_loc)
        
        self.code = f"""
                          
                        {self.headers}
                        
                        extern "C" __global__ void GpuConv1DOnDevice(int nx, int ny, {dtype} *out, {dtype} **{arg.id}) {{
    
                          // get the index of the current thread
                          int i = blockIdx.x * blockDim.x + threadIdx.x;

                          // declare shared mem
                          extern __shared__ {dtype} yj[];
            
                          // load parameter(s)
                          {param_loc.declare()}
                          {load_vars(dimsp, indsp, param_loc, args)}
                          
                          {fout.declare()}
    
                          // get the value of variable (index with i)
                          {xi.declare()}
                          if (i < nx) {{
                              {load_vars(dimsx, indsi, xi, args, row_index=i)} // load xi variables from global memory to local thread memory
                              #pragma unroll
                              for (int k=0; k<{dimout}; k++) {{
                                  out[i*{dimout}+k] = 0.0f;
                              }}
                          }}
                          
                          {acc.declare()}

                          for (int jstart = 0, tile = 0; jstart < ny; jstart += blockDim.x, tile++) {{

                              // get the current column
                              int j = tile * blockDim.x + threadIdx.x;

                              if (j < ny) {{ // we load yj from device global memory only if j<ny
                                  {load_vars(dimsy, indsj, yjloc, args, row_index=j)} // load yj variables from global memory to shared memory
                              }}
                              __syncthreads();

                              if (i < nx) {{ // we compute x1i only if needed
                                  TYPE * yjrel = yj; // Loop on the columns of the current block.
                                  for (int jrel = 0; (jrel < {blocksize_chunks}) && (jrel < ny - jstart); jrel++, yjrel += {dimy}) {{
                                      {formula(foutjrel, table)} // Call the function, which outputs results in fout
                                  }}
                              }}
        
                              __syncthreads();
        
                              for (int chunk=0; chunk<{nchunks-1}; chunk++) {{
                                  do_finalchunk_sub(dtype, varfinal, dimfinalchunk,
                                                    acc, i, j, jstart, chunk, nx, ny, arg, fout, yj, out)
                              }}
                              do_finalchunk_sub(dtype, varfinal, dimlastfinalchunk,
                                                acc, i, j, jstart, {nchunks-1}, nx, ny, arg, fout, yj, out)
                          }}
                        }}
                    """
