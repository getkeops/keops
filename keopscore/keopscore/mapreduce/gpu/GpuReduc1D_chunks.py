from keopscore import cuda_block_size
from keopscore.config.chunks import dimchunk
from keopscore.binders.nvrtc.Gpu_link_compile import Gpu_link_compile
from keopscore.formulas.reductions.sum_schemes import *
from keopscore.mapreduce.gpu.GpuAssignZero import GpuAssignZero
from keopscore.mapreduce.MapReduce import MapReduce
from keopscore.utils.code_gen_utils import (
    load_vars,
    load_vars_chunks,
    sizeof,
    pointer,
    table,
    table4,
    use_pragma_unroll,
)
from keopscore.mapreduce.Chunk_Mode_Constants import Chunk_Mode_Constants


def do_chunk_sub(
    dtype,
    red_formula,
    fun_chunked_curr,
    dimchunk_curr,
    dimsx,
    dimsy,
    indsi,
    indsj,
    indsi_chunked,
    indsj_chunked,
    acc,
    tile,
    i,
    j,
    jstart,
    chunk,
    nx,
    ny,
    arg,
    fout,
    xi,
    yj,
    param_loc,
):
    chk = Chunk_Mode_Constants(red_formula)
    fout_tmp_chunk = c_array(dtype, chk.fun_chunked.dim)
    xiloc = c_variable(pointer(dtype), f"({xi.id} + {chk.dimx_notchunked})")
    yjloc = c_variable(
        pointer(dtype), f"({yj.id} + threadIdx.x * {chk.dimy} + {chk.dimy_notchunked})"
    )
    load_chunks_routine_i = load_vars_chunks(
        indsi_chunked,
        dimchunk,
        dimchunk_curr,
        chk.dim_org,
        xiloc,
        arg,
        chunk,
        row_index=i,
    )
    load_chunks_routine_j = load_vars_chunks(
        indsj_chunked,
        dimchunk,
        dimchunk_curr,
        chk.dim_org,
        yjloc,
        arg,
        chunk,
        row_index=j,
    )
    yjrel = c_variable(pointer(dtype), "yjrel")
    chktable = table(
        chk.nminargs,
        dimsx,
        dimsy,
        chk.dimsp,
        indsi,
        indsj,
        chk.indsp,
        xi,
        yjrel,
        param_loc,
    )
    foutj = c_variable(pointer(dtype), "foutj")

    return f"""
                // Starting chunk_sub routine
                {fout_tmp_chunk.declare()}
                if ({i.id} < {nx.id}) {{
                    {load_chunks_routine_i}
                }} 
                if (j < ny) {{ // we load yj from device global memory only if j<ny
                    {load_chunks_routine_j}
                }}
                __syncthreads();
                if ({i.id} < {nx.id}) {{ // we compute only if needed
                    {dtype} *yjrel = {yj.id}; // Loop on the columns of the current block.
                    for (int jrel = 0; (jrel < blockDim.x) && (jrel < {ny.id} - jstart); jrel++, yjrel += {chk.dimy}) {{
                        {dtype} *foutj = {fout.id} + jrel*{chk.fun_chunked.dim};
                        {fun_chunked_curr(fout_tmp_chunk, chktable)}
                        {chk.fun_chunked.acc_chunk(foutj, fout_tmp_chunk)}
                    }}
                }} 
                __syncthreads();
                // Finished chunk_sub routine
            """


class GpuReduc1D_chunks(MapReduce, Gpu_link_compile):
    # class for generating the final C++ code, Gpu version

    AssignZero = GpuAssignZero

    def __init__(self, *args):
        MapReduce.__init__(self, *args)
        Gpu_link_compile.__init__(self)
        self.chk = Chunk_Mode_Constants(self.red_formula)
        self.dimy = self.chk.dimy
        self.blocksize_chunks = min(
            cuda_block_size, 1024, 49152 // max(1, self.dimy * sizeof(self.dtype))
        )

    def get_code(self):
        super().get_code()

        red_formula = self.red_formula
        dtype = self.dtype
        dtypeacc = self.dtypeacc
        varloader = self.varloader

        i = self.i
        j = self.j

        arg = self.arg
        args = self.args

        yjrel = c_array(dtype, varloader.dimy, "yjrel")

        jreltile = c_variable("int", "(jrel + tile * blockDim.x)")

        chk = self.chk
        param_loc = c_array(dtype, chk.dimp, "param_loc")
        acc = c_array(dtypeacc, chk.dimred, "acc")
        sum_scheme = eval(self.sum_scheme_string)(red_formula, dtype, dimred=chk.dimred)
        xi = c_array(dtype, chk.dimx, "xi")
        fout_chunk = c_array(
            dtype, self.blocksize_chunks * chk.dimout_chunk, "fout_chunk"
        )
        yj = c_variable(pointer(dtype), "yj")
        yjloc = c_array(dtype, chk.dimy, f"(yj + threadIdx.x * {chk.dimy})")

        fout_chunk_loc = c_variable(
            pointer(dtype), f"({fout_chunk.id}+jrel*{chk.dimout_chunk})"
        )

        tile = c_variable("int", "tile")
        nx = c_variable("int", "nx")
        ny = c_variable("int", "ny")

        jstart = c_variable("int", "jstart")
        chunk = c_variable("int", "chunk")

        chunk_sub_routine = do_chunk_sub(
            dtype,
            red_formula,
            chk.fun_chunked,
            dimchunk,
            chk.dimsx,
            chk.dimsy,
            chk.indsi,
            chk.indsj,
            chk.indsi_chunked,
            chk.indsj_chunked,
            acc,
            tile,
            i,
            j,
            jstart,
            chunk,
            nx,
            ny,
            arg,
            fout_chunk,
            xi,
            yj,
            param_loc,
        )

        last_chunk = c_variable("int", f"{chk.nchunks - 1}")
        chunk_sub_routine_last = do_chunk_sub(
            dtype,
            red_formula,
            chk.fun_lastchunked,
            chk.dimlastchunk,
            chk.dimsx_last,
            chk.dimsy_last,
            chk.indsi,
            chk.indsj,
            chk.indsi_lastchunked,
            chk.indsj_lastchunked,
            acc,
            tile,
            i,
            j,
            jstart,
            last_chunk,
            nx,
            ny,
            arg,
            fout_chunk,
            xi,
            yj,
            param_loc,
        )

        foutj = c_array(dtype, chk.dimout_chunk, "foutj")
        chktable_out = table4(
            chk.nminargs + 1,
            chk.dimsx,
            chk.dimsy,
            chk.dimsp,
            [chk.dimout_chunk],
            chk.indsi,
            chk.indsj,
            chk.indsp,
            [chk.nminargs],
            xi,
            yjrel,
            param_loc,
            foutj,
        )
        fout_tmp = c_array(dtype, chk.dimfout, "fout_tmp")
        outi = c_array(dtype, chk.dimout, f"(out + i * {chk.dimout})")

        self.code = f"""
                          
                        {self.headers}
                        
                        extern "C" __global__ void GpuConv1DOnDevice(int nx, int ny, {dtype} *out, {dtype} **{arg.id}) {{
    
                          // get the index of the current thread
                          int i = blockIdx.x * blockDim.x + threadIdx.x;

                          // declare shared mem
                          extern __shared__ {dtype} yj[];

                          // load parameters variables from global memory to local thread memory
                          {param_loc.declare()}
                          {load_vars(chk.dimsp, chk.indsp, param_loc, args)}
                          
                          {acc.declare()}
                          
                          {sum_scheme.declare_temporary_accumulator()}                     
                          
                          if (i < nx) {{
                            {red_formula.InitializeReduction(acc)} // acc = 0
                            {sum_scheme.initialize_temporary_accumulator_first_init()}
                          }}

                          {xi.declare()}

                          {fout_chunk.declare()}
                          
                          if (i < nx) {{
                            {load_vars(chk.dimsx_notchunked, chk.indsi_notchunked, xi, args, row_index=i)} // load xi variables from global memory to local thread memory
                          }}

                          for (int jstart = 0, tile = 0; jstart < ny; jstart += blockDim.x, tile++) {{

                            // get the current column
                            int j = tile * blockDim.x + threadIdx.x;

                            if (j < ny) {{ // we load yj from device global memory only if j<ny
                              {load_vars(chk.dimsy_notchunked, chk.indsj_notchunked, yjloc, args, row_index=j)} 
                            }}
                            __syncthreads();

                            if (i < nx) {{ // we compute x1i only if needed
                              for (int jrel = 0; (jrel < blockDim.x) && (jrel < ny - jstart); jrel++) {{
                                {chk.fun_chunked.initacc_chunk(fout_chunk_loc)}
                              }}
                              {sum_scheme.initialize_temporary_accumulator_block_init()}
                            }}
                            
                            // looping on chunks (except the last)
                    		{use_pragma_unroll()}
                    		for (int chunk=0; chunk<{chk.nchunks}-1; chunk++) {{
                              {chunk_sub_routine}
                            }}
                            // last chunk
                            {chunk_sub_routine_last}
                            
                            if (i < nx) {{
                                {dtype} * yjrel = yj; // Loop on the columns of the current block.
                                for (int jrel = 0; (jrel < blockDim.x) && (jrel < ny - jstart); jrel++, yjrel += {chk.dimy}) {{
                                    {dtype} *foutj = fout_chunk + jrel*{chk.dimout_chunk};
                                    {fout_tmp.declare()}
                                    {chk.fun_postchunk(fout_tmp, chktable_out)}
                                    {sum_scheme.accumulate_result(acc, fout_tmp, jreltile)}
                                }}
                                {sum_scheme.final_operation(acc)}
                            }}
                            __syncthreads();
                          }}

                          if (i < nx) {{
                            {red_formula.FinalizeOutput(acc, outi, i)} 
                          }}
                        }}
                    """
