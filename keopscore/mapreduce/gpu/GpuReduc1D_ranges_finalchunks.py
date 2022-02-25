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
    load_vars_chunks_offsets,
    sizeof,
    pointer,
    Var_loader,
    use_pragma_unroll,
)
from keopscore.utils.misc_utils import KeOps_Error


def do_finalchunk_sub_ranges(
    dtype,
    fun_global,
    varfinal,
    dimfinalchunk_curr,
    acc,
    i,
    j,
    jstart,
    start_y,
    chunk,
    end_x,
    end_y,
    nbatchdims,
    indices_j,
    arg,
    fout,
    yj,
    out,
):

    dimout = varfinal.dim
    yjloc = c_variable(pointer(dtype), f"({yj.id} + threadIdx.x * {dimfinalchunk})")
    indsj_global = Var_loader(fun_global).indsj
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
    load_chunks_routine_j_ranges = load_vars_chunks_offsets(
        [varfinal.ind],
        indsj_global,
        dimfinalchunk,
        dimfinalchunk_curr,
        varfinal.dim,
        yjloc,
        arg,
        chunk,
        indices_j,
        row_index=j - start_y,
    )
    return f"""
                {acc.assign(c_zero_float)}
                {dtype} *yjrel = yj;
                if ({j.id} < {end_y.id}) {{ // we load yj from device global memory only if j<end_y
                    if ({nbatchdims.id}==0) {{
                        {load_chunks_routine_j}
                    }} else {{
                        {load_chunks_routine_j_ranges}
                    }}
                }}
                __syncthreads();
                for (int jrel = 0; (jrel < blockDim.x) && (jrel < {end_y.id} - {jstart.id}); jrel++, yjrel += {dimfinalchunk}) {{          
                    if ({i.id} < {end_x.id}) {{ // we compute only if needed
                        {use_pragma_unroll()}
                        for (int k=0; k<{dimfinalchunk_curr}; k++) {{
                            {acc.id}[k] += yjrel[k] * fout[jrel];
                        }}
                    }}
                    __syncthreads();
                }}
                if ({i.id} < {end_x.id}) {{
                    {use_pragma_unroll()}
                    for (int k=0; k<{dimfinalchunk_curr}; k++)
                        {out.id}[i*{dimout}+{chunk.id}*{dimfinalchunk}+k] += {acc.id}[k];
                }}
                __syncthreads();
            """


class GpuReduc1D_ranges_finalchunks(MapReduce, Gpu_link_compile):
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
        table = varloader.table(xi, yjrel, param_loc)

        lastchunk = c_variable("int", f"{nchunks-1}")

        startx = c_variable("int", "start_x")
        starty = c_variable("int", "start_y")

        end_x = c_variable("int", "end_x")
        end_y = c_variable("int", "end_y")

        nbatchdims = c_variable("int", "nbatchdims")

        fun_global = self.red_formula
        varloader_global = Var_loader(fun_global)
        indsi_global = varloader_global.indsi
        indsj_global = varloader_global.indsj
        nvarsi_global, nvarsj_global, nvarsp_global = (
            len(varloader_global.Varsi),
            len(varloader_global.Varsj),
            len(varloader_global.Varsp),
        )
        nvars_global = nvarsi_global + nvarsj_global + nvarsp_global
        offsets = c_array("int", nvars_global, "offsets")

        indices_i = c_array("int", nvarsi_global, "indices_i")
        indices_j = c_array("int", nvarsj_global, "indices_j")
        indices_p = c_array("int", nvarsp_global, "indices_p")

        declare_assign_indices_i = (
            "int *indices_i = offsets;" if nvarsi_global > 0 else ""
        )
        declare_assign_indices_j = (
            f"int *indices_j = offsets + {nvarsi_global};" if nvarsj_global > 0 else ""
        )
        declare_assign_indices_p = (
            f"int *indices_p = offsets + {nvarsi_global} + {nvarsj_global};"
            if nvarsp_global > 0
            else ""
        )

        chunk_sub_routine = do_finalchunk_sub_ranges(
            dtype,
            fun_global,
            varfinal,
            dimfinalchunk,
            acc,
            i,
            j,
            jstart,
            starty,
            chunk,
            end_x,
            end_y,
            nbatchdims,
            indices_j,
            arg,
            fout,
            yj,
            out,
        )

        chunk_sub_routine_last = do_finalchunk_sub_ranges(
            dtype,
            fun_global,
            varfinal,
            dimlastfinalchunk,
            acc,
            i,
            j,
            jstart,
            starty,
            lastchunk,
            end_x,
            end_y,
            nbatchdims,
            indices_j,
            arg,
            fout,
            yj,
            out,
        )

        threadIdx_x = c_variable("int", "threadIdx.x")

        self.code = f"""
                          
                        {self.headers}
                        
                        extern "C" __global__ void GpuConv1DOnDevice_ranges(int nx, int ny, int nbatchdims,
                                                    int *offsets_d, int *lookup_d, int *slices_x,
                                                    int *ranges_y, {dtype} *out, {dtype} **{arg.id}) {{
                                                        
                          {offsets.declare()}
                          {declare_assign_indices_i}
                          {declare_assign_indices_j}
                          {declare_assign_indices_p}
                          
                          if (nbatchdims > 0) {{
                              for (int k = 0; k < {nvars_global}; k++) {{
                                  offsets[k] = offsets_d[ {nvars_global} * blockIdx.x + k ];
                              }}
                          }}
                       
                          // Retrieve our position along the laaaaarge [1,~nx] axis: -----------------
                          int range_id= (lookup_d)[3*blockIdx.x] ;
                          int start_x = (lookup_d)[3*blockIdx.x+1] ;
                          int end_x   = (lookup_d)[3*blockIdx.x+2] ;
                          
                          // The "slices_x" vector encodes a set of cutting points in
                          // the "ranges_y" array of ranges.
                          // As discussed in the Genred docstring, the first "0" is implicit:
                          int start_slice = range_id < 1 ? 0 : slices_x[range_id-1];
                          int end_slice   = slices_x[range_id];
                          
                          // get the index of the current thread
                          int i = start_x + threadIdx.x;
                          
                          // declare shared mem
                          extern __shared__ {dtype} yj[];
                          
                          // load parameter(s)
                          {param_loc.declare()}
                          {load_vars(dimsp, indsp, param_loc, args)}
                          
                          {fout.declare()}
    
                          // get the value of variable (index with i)
                          {xi.declare()}
                          if (i < end_x) {{
                              
                              if (nbatchdims == 0) {{
                                  {varloader.load_vars("i", xi, args, row_index=i)} // load xi variables from global memory to local thread memory
                              }} else {{
                                  {varloader.load_vars("i", xi, args, row_index=threadIdx_x, offsets=indices_i, indsref=indsi_global)} // Possibly, with offsets as we support broadcasting over batch dimensions
                              }}
                              
                              {use_pragma_unroll()}
                              for (int k=0; k<{dimout}; k++) {{
                                  out[i*{dimout}+k] = 0.0f;
                              }}
                          }}
                          
                          {acc.declare()}
                          
                          int start_y = ranges_y[2*start_slice], end_y = 0;
                          for(int index = start_slice ; index < end_slice ; index++ ) {{
                              if( (index+1 >= end_slice) || (ranges_y[2*index+2] != ranges_y[2*index+1]) ) {{
                                  end_y = ranges_y[2*index+1];
                                  for (int jstart = start_y, tile = 0; jstart < end_y; jstart += blockDim.x, tile++) {{
                                      
                                      // get the current column
                                      int j = jstart + threadIdx.x;
                                      
                                      if (j < end_y) {{ // we load yj from device global memory only if j<end_y
                                          if (nbatchdims == 0) {{
                                              {varloader.load_vars("j", yjloc, args, row_index=j)} // load yj variables from global memory to shared memory
                                          }} else {{
                                              {varloader.load_vars("j", yjloc, args, row_index=j-starty, offsets=indices_j, indsref=indsj_global)} // Possibly, with offsets as we support broadcasting over batch dimensions
                                          }}
                                      }}
                                      __syncthreads();
                                      
                                      if (i < end_x) {{ // we compute x1i only if needed
                                          {dtype} * yjrel = yj; // Loop on the columns of the current block.
                                          for (int jrel = 0; (jrel < {blocksize_chunks}) && (jrel < end_y - jstart); jrel++, yjrel += {dimy}) {{
                                              {formula(foutjrel, table)} // Call the function, which outputs results in fout
                                          }}
                                      }}
                                      __syncthreads();
                                      
                                      for (int chunk=0; chunk<{nchunks-1}; chunk++) {{
                                          {chunk_sub_routine}
                                      }}
                                      {chunk_sub_routine_last}
                                  }}
                                  if(index+1 < end_slice) 
                                      start_y = ranges_y[2*index+2];
                              }}
                          }}
                       }}
                    """
