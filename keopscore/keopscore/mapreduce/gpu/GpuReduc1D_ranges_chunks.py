from keopscore import cuda_block_size
from keopscore.config.chunks import dimchunk
from keopscore.binders.nvrtc.Gpu_link_compile import Gpu_link_compile
from keopscore.formulas.reductions.sum_schemes import *
from keopscore.mapreduce.gpu.GpuAssignZero import GpuAssignZero
from keopscore.mapreduce.MapReduce import MapReduce
from keopscore.utils.code_gen_utils import (
    load_vars,
    load_vars_chunks,
    load_vars_chunks_offsets,
    sizeof,
    pointer,
    table,
    table4,
    Var_loader,
    use_pragma_unroll,
)
from keopscore.mapreduce.Chunk_Mode_Constants import Chunk_Mode_Constants


def do_chunk_sub_ranges(
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
    start_y,
    chunk,
    end_x,
    end_y,
    nbatchdims,
    indices_i,
    indices_j,
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

    varloader_global = Var_loader(red_formula)
    indsi_global = varloader_global.indsi
    indsj_global = varloader_global.indsj

    threadIdx_x = c_variable("int", "threadIdx.x")
    load_chunks_routine_i_batches = load_vars_chunks_offsets(
        indsi_chunked,
        indsi_global,
        dimchunk,
        dimchunk_curr,
        chk.dim_org,
        xiloc,
        arg,
        chunk,
        indices_i,
        row_index=threadIdx_x,
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

    load_chunks_routine_j_batches = load_vars_chunks_offsets(
        indsj_chunked,
        indsj_global,
        dimchunk,
        dimchunk_curr,
        chk.dim_org,
        yjloc,
        arg,
        chunk,
        indices_j,
        row_index=j - start_y,
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
                {fout_tmp_chunk.declare()}
                if ({i.id} < {end_x.id}) {{
                    if ({nbatchdims.id}==0) {{
                        {load_chunks_routine_i}
                    }} else {{
                        {load_chunks_routine_i_batches}
                    }}
                }} 
                if ({j.id} < {end_y.id}) {{ // we load yj from device global memory only if j<ny
                    if ({nbatchdims.id}==0) {{
                        {load_chunks_routine_j}
                    }} else {{
                        {load_chunks_routine_j_batches}
                    }}
                }}
                __syncthreads();
                if ({i.id} < {end_x.id}) {{ // we compute only if needed
            		{dtype} *yjrel = {yj.id}; // Loop on the columns of the current block.
            		for (int jrel = 0; (jrel < blockDim.x) && (jrel < {end_y.id} - jstart); jrel++, yjrel += {chk.dimy}) {{
            			{dtype} *foutj = {fout.id} + jrel*{chk.fun_chunked.dim};
                        {fun_chunked_curr(fout_tmp_chunk, chktable)}
                        {chk.fun_chunked.acc_chunk(foutj, fout_tmp_chunk)}
            		}}
                }} 
                __syncthreads();
            """


class GpuReduc1D_ranges_chunks(MapReduce, Gpu_link_compile):
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

        varloader_global = Var_loader(red_formula)
        indsi_global = varloader_global.indsi
        indsj_global = varloader_global.indsj

        i = self.i
        j = self.j

        arg = self.arg
        args = self.args

        nvarsi, nvarsj, nvarsp = (
            len(self.varloader.Varsi),
            len(self.varloader.Varsj),
            len(self.varloader.Varsp),
        )
        nvars = nvarsi + nvarsj + nvarsp

        indices_i = c_array("int", nvarsi, "indices_i")
        indices_j = c_array("int", nvarsj, "indices_j")
        indices_p = c_array("int", nvarsp, "indices_p")

        declare_assign_indices_i = "int *indices_i = offsets;" if nvarsi > 0 else ""
        declare_assign_indices_j = (
            f"int *indices_j = offsets + {nvarsi};" if nvarsj > 0 else ""
        )
        declare_assign_indices_p = (
            f"int *indices_p = offsets + {nvarsi} + {nvarsj};" if nvarsp > 0 else ""
        )

        yjrel = c_array(dtype, varloader_global.dimy, "yjrel")

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

        end_x = c_variable("int", "end_x")
        end_y = c_variable("int", "end_y")

        starty = c_variable("int", "start_y")

        nbatchdims = c_variable("int", "nbatchdims")

        chunk_sub_routine = do_chunk_sub_ranges(
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
            starty,
            chunk,
            end_x,
            end_y,
            nbatchdims,
            indices_i,
            indices_j,
            arg,
            fout_chunk,
            xi,
            yj,
            param_loc,
        )

        last_chunk = c_variable("int", f"{chk.nchunks-1}")
        chunk_sub_routine_last = do_chunk_sub_ranges(
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
            starty,
            last_chunk,
            end_x,
            end_y,
            nbatchdims,
            indices_i,
            indices_j,
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

        threadIdx_x = c_variable("int", "threadIdx.x")

        self.code = f"""
                          
                        {self.headers}
                        
                        extern "C" __global__ void GpuConv1DOnDevice_ranges(int nx, int ny, int nbatchdims,
                                                    int *offsets_d, int *lookup_d, int *slices_x,
                                                    int *ranges_y, {dtype} *out, {dtype} **{arg.id}) {{
                                                        
                          int offsets[{nvars}];
                          {declare_assign_indices_i}
                          {declare_assign_indices_j}
                          {declare_assign_indices_p}
                          
                          if (nbatchdims > 0)
                              for (int k = 0; k < {nvars}; k++)
                                  offsets[k] = offsets_d[ {nvars} * blockIdx.x + k ];
                                  
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

                          // load parameters variables from global memory to local thread memory
                          {param_loc.declare()}
                          if (nbatchdims == 0) {{
                              {load_vars(chk.dimsp, chk.indsp, param_loc, args)}
                          }} else {{
                              {load_vars(chk.dimsp, chk.indsp, param_loc, args, offsets=indices_p)}
                          }}
                          
                          {acc.declare()}
                          
                          {sum_scheme.declare_temporary_accumulator()}                     
                          
                          if (i < end_x) {{
                            {red_formula.InitializeReduction(acc)} // acc = 0
                            {sum_scheme.initialize_temporary_accumulator_first_init()}
                          }}

                          {xi.declare()}

                          {fout_chunk.declare()}
                          
                          if (i < end_x) {{
                              // load xi variables from global memory to local thread memory
                              if (nbatchdims == 0) {{
                                  {load_vars(chk.dimsx_notchunked, chk.indsi_notchunked, xi, args, row_index=i)} 
                              }} else {{
                                  {load_vars(chk.dimsx_notchunked, chk.indsi_notchunked, xi, args, 
                                              row_index=threadIdx_x, offsets=indices_i, indsref=indsi_global)}
                              }} 
                          }}
                          
                          int start_y = ranges_y[2*start_slice], end_y = 0;
                          for( int index = start_slice ; index < end_slice ; index++ ) {{
                              if( (index+1 >= end_slice) || (ranges_y[2*index+2] != ranges_y[2*index+1]) ) {{
                                  //start_y = ranges_y[2*index] ;
                                  end_y = ranges_y[2*index+1];
                                  
                                  for(int jstart = start_y, tile = 0; jstart < end_y; jstart += blockDim.x, tile++) {{
                                      // get the current column
                                      int j = jstart + threadIdx.x;
                                      
                                      if(j<end_y) // we load yj from device global memory only if j<end_y
                                          if (nbatchdims == 0) {{
                                              // load yj variables from global memory to shared memory
                                              {load_vars(chk.dimsy_notchunked, chk.indsj_notchunked, yjloc, args, row_index=j)} 
                                          }} else {{
                                              // Possibly, with offsets as we support broadcasting over batch dimensions
                                              {load_vars(chk.dimsy_notchunked, chk.indsj_notchunked, yjloc, args, 
                                                          row_index=j-starty, offsets=indices_j, indsref=indsj_global)}
                                          }}
                                      __syncthreads();
                                      
                                      if(i<end_x) {{ // we compute x1i only if needed
                                      for (int jrel = 0; (jrel < blockDim.x) && (jrel < end_y - jstart); jrel++) {{
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
                                  
                                  if (i < end_x) {{
                                      {dtype} * yjrel = yj; // Loop on the columns of the current block.
                                      if (nbatchdims == 0) {{
                                          for (int jrel = 0; (jrel < blockDim.x) && (jrel <end_y - jstart); jrel++, yjrel += {chk.dimy}) {{
                                              {dtype} *foutj = fout_chunk + jrel*{chk.dimout_chunk};
                                              {fout_tmp.declare()}
                                              {chk.fun_postchunk(fout_tmp, chktable_out)}
                                              {sum_scheme.accumulate_result(acc, fout_tmp, jreltile)}
                                          }}
                                      }} else {{
                                          for (int jrel = 0; (jrel < blockDim.x) && (jrel <end_y - jstart); jrel++, yjrel += {chk.dimy}) {{
                                              {dtype} *foutj = fout_chunk + jrel*{chk.dimout_chunk};
                                              {fout_tmp.declare()}
                                              {chk.fun_postchunk(fout_tmp, chktable_out)}
                                              {sum_scheme.accumulate_result(acc, fout_tmp, jreltile)}
                                          }}
                                      }}
                                      {sum_scheme.final_operation(acc)}
                                  }}
                                  __syncthreads();
                              }}
                              
                              if(index+1 < end_slice) {{
                                  start_y = ranges_y[2*index+2] ;
                  			  }}
                          }}
                      }}
                      if (i < end_x) {{
                          {red_formula.FinalizeOutput(acc, outi, i)} 
                      }}
                  }}
                    """
