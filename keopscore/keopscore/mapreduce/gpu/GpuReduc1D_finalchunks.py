from keopscore import cuda_block_size
from keopscore.config.chunks import dimfinalchunk
from keopscore.binders.nvrtc.Gpu_link_compile import Gpu_link_compile
from keopscore.formulas.reductions.Sum_Reduction import Sum_Reduction
from keopscore.formulas.reductions.sum_schemes import *
from keopscore.mapreduce.gpu.GpuAssignZero import GpuAssignZero
from keopscore.mapreduce.MapReduce import MapReduce
from keopscore.utils.code_gen_utils import (
    load_vars,
    load_vars_chunks,
    sizeof,
    pointer,
    Var_loader,
    use_pragma_unroll,
)
from keopscore.utils.misc_utils import KeOps_Error


def do_finalchunk_sub(
    dtype,
    varfinal,
    dimfinalchunk_curr,
    acc,
    i,
    j,
    jstart,
    chunk,
    nx,
    ny,
    arg,
    fout,
    yj,
    out,
):

    dimout = varfinal.dim
    yjloc = c_variable(pointer(dtype), f"({yj.id} + threadIdx.x * {dimfinalchunk})")
    load_chunks_routine_j = load_vars_chunks(
        [varfinal.ind],
        dimfinalchunk,
        dimfinalchunk_curr,
        varfinal.dim,
        yjloc,
        arg,
        chunk,
        row_index=j,
    )
    return f"""
                {acc.assign(c_zero_float)}
                {dtype} *yjrel = yj;
                if ({j.id} < {ny.id}) {{ // we load yj from device global memory only if j<ny
                    {load_chunks_routine_j}
                }}
                __syncthreads();
                for (int jrel = 0; (jrel < blockDim.x) && (jrel < {ny.id} - {jstart.id}); jrel++, yjrel += {dimfinalchunk}) {{          
                    if ({i.id} < {nx.id}) {{ // we compute only if needed
                        {use_pragma_unroll()}
                        for (int k=0; k<{dimfinalchunk_curr}; k++) {{
                            {acc.id}[k] += yjrel[k] * fout[jrel];
                        }}
                    }}
                    __syncthreads();
                }}
                if ({i.id} < {nx.id}) {{
                    {use_pragma_unroll()}
                    for (int k=0; k<{dimfinalchunk_curr}; k++)
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

    def get_code(self):

        super().get_code()
        dtype = self.dtype
        dtypeacc = self.dtypeacc
        i = self.i
        j = self.j
        nx = c_variable("int", "nx")
        ny = c_variable("int", "ny")
        jstart = c_variable("int", "jstart")
        chunk = c_variable("int", "chunk")
        arg = self.arg
        args = self.args
        yj = c_variable(pointer(dtype), "yj")
        out = c_variable(pointer(dtype), "out")

        fun_internal = Sum_Reduction(
            self.red_formula.formula.children[0], self.red_formula.tagI
        )
        formula = fun_internal.formula

        varfinal = self.red_formula.formula.children[1]
        nchunks = 1 + (varfinal.dim - 1) // dimfinalchunk
        dimlastfinalchunk = varfinal.dim - (nchunks - 1) * dimfinalchunk
        varloader = Var_loader(fun_internal)
        dimsx = varloader.dimsx
        dimsy = varloader.dimsy
        dimsp = varloader.dimsp
        indsi = varloader.indsi
        indsj = varloader.indsj
        indsp = varloader.indsp
        dimx = sum(dimsx)
        dimy = sum(dimsy)
        dimp = sum(dimsp)
        dimout = varfinal.dim
        dimfout = fun_internal.formula.dim
        if dimfout != 1:
            KeOps_Error("dimfout should be 1")
        sum_scheme = self.sum_scheme

        self.dimy = max(dimfinalchunk, dimy)
        blocksize_chunks = min(
            cuda_block_size, 1024, 49152 // max(1, self.dimy * sizeof(self.dtype))
        )

        if not isinstance(sum_scheme, block_sum):
            KeOps_Error("only block_sum available")
        param_loc = c_array(dtype, dimp, "param_loc")
        fout = c_array(dtype, dimfout * blocksize_chunks, "fout")
        xi = c_array(dtype, dimx, "xi")
        acc = c_array(dtypeacc, dimfinalchunk, "acc")
        yjloc = c_array(dtype, dimy, f"(yj + threadIdx.x * {dimy})")
        foutjrel = c_array(dtype, dimfout, f"({fout.id}+jrel*{dimfout})")
        yjrel = c_array(dtype, dimy, "yjrel")
        table = self.varloader.table(xi, yjrel, param_loc)

        last_chunk = c_variable("int", f"{nchunks-1}")

        chunk_sub_routine = do_finalchunk_sub(
            dtype,
            varfinal,
            dimfinalchunk,
            acc,
            i,
            j,
            jstart,
            chunk,
            nx,
            ny,
            arg,
            fout,
            yj,
            out,
        )

        chunk_sub_routine_last = do_finalchunk_sub(
            dtype,
            varfinal,
            dimlastfinalchunk,
            acc,
            i,
            j,
            jstart,
            last_chunk,
            nx,
            ny,
            arg,
            fout,
            yj,
            out,
        )

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
                              {use_pragma_unroll()}
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
                                  {dtype} * yjrel = yj; // Loop on the columns of the current block.
                                  for (int jrel = 0; (jrel < {blocksize_chunks}) && (jrel < ny - jstart); jrel++, yjrel += {dimy}) {{
                                      {formula(foutjrel, table)} // Call the function, which outputs results in fout
                                  }}
                              }}
        
                              __syncthreads();
        
                              for (int chunk=0; chunk<{nchunks-1}; chunk++) {{
                                  {chunk_sub_routine}
                              }}
                              {chunk_sub_routine_last}
                          }}
                        }}
                    """
