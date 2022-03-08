from keopscore.binders.nvrtc.Gpu_link_compile import Gpu_link_compile
from keopscore.mapreduce.gpu.GpuAssignZero import GpuAssignZero
from keopscore.mapreduce.MapReduce import MapReduce
from keopscore.utils.code_gen_utils import (
    c_variable,
    c_array,
    c_include,
)


class GpuReduc1D_ranges(MapReduce, Gpu_link_compile):
    # class for generating the final C++ code, Gpu version

    AssignZero = GpuAssignZero

    def __init__(self, *args):
        MapReduce.__init__(self, *args)
        Gpu_link_compile.__init__(self)
        self.dimy = self.varloader.dimy

    def get_code(self):

        super().get_code()

        red_formula = self.red_formula
        dtype = self.dtype
        varloader = self.varloader

        i = self.i
        j = self.j
        fout = self.fout
        outi = self.outi
        acc = self.acc
        arg = self.arg
        args = self.args
        sum_scheme = self.sum_scheme

        nvarsi, nvarsj, nvarsp = (
            len(self.varloader.Varsi),
            len(self.varloader.Varsj),
            len(self.varloader.Varsp),
        )
        nvars = nvarsi + nvarsj + nvarsp

        param_loc = self.param_loc
        xi = self.xi
        yjloc = c_array(dtype, varloader.dimy, f"(yj + threadIdx.x * {varloader.dimy})")
        yjrel = c_array(dtype, varloader.dimy, "yjrel")
        table = varloader.table(self.xi, yjrel, self.param_loc)
        jreltile = c_variable("int", "(jrel + tile * blockDim.x)")

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

        starty = c_variable("int", "start_y")

        threadIdx_x = c_variable("int", "threadIdx.x")

        if dtype == "half2":
            self.headers += c_include("cuda_fp16.h")

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
                          
                          // load parameter(s)
                          {param_loc.declare()}

                          if (nbatchdims == 0) {{
                              {varloader.load_vars("p", param_loc, args)}
                          }} else {{
                              {varloader.load_vars("p", param_loc, args, offsets=indices_p)}
                          }}
                          {fout.declare()}
                          {xi.declare()}
                          {acc.declare()}
                          {sum_scheme.declare_temporary_accumulator()}

                          if(i<end_x) {{
                              {red_formula.InitializeReduction(acc)} // acc = 0
                              {sum_scheme.initialize_temporary_accumulator_first_init()}
                              if (nbatchdims == 0) {{
                                  {varloader.load_vars('i', xi, args, row_index=i)} // load xi variables from global memory to local thread memory
                              }} else {{
                                  {varloader.load_vars('i', xi, args, row_index=threadIdx_x, offsets=indices_i)}  // Possibly, with offsets as we support broadcasting over batch dimensions
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
                                              {varloader.load_vars('j', yjloc, args, row_index=j)} // load yj variables from global memory to shared memory
                                          }} else {{
                                              {varloader.load_vars('j', yjloc, args, row_index=j-starty, offsets=indices_j)}  // Possibly, with offsets as we support broadcasting over batch dimensions
                                          }}
                                      __syncthreads();
                                      
                                      if(i<end_x) {{ // we compute x1i only if needed
                                          {dtype} * yjrel = yj; // Loop on the columns of the current block.
                                          {sum_scheme.initialize_temporary_accumulator_block_init()}
                                          if (nbatchdims == 0) {{
                                              for(int jrel = 0; (jrel < blockDim.x) && (jrel<end_y-jstart); jrel++, yjrel+={varloader.dimy}) {{
                                                  {red_formula.formula(fout,table)} // Call the function, which outputs results in xi[0:DIMX1]
                                                  {sum_scheme.accumulate_result(acc, fout, jreltile+starty)}
                                              }} 
                                          }} else {{
                                              for(int jrel = 0; (jrel < blockDim.x) && (jrel<end_y-jstart); jrel++, yjrel+={varloader.dimy}) {{
                                                  {red_formula.formula(fout,table)} // Call the function, which outputs results in fout
                                                  {sum_scheme.accumulate_result(acc, fout, jreltile)}
                                              }}
                                          }}
                                          {sum_scheme.final_operation(acc)}
                                      }}
                                      __syncthreads();
                                  }}
                                  if(index+1 < end_slice)
                                      start_y = ranges_y[2*index+2] ;
                              }}
                          }}
                          if(i<end_x) {{
                          	{red_formula.FinalizeOutput(acc, outi, i)} 
                          }}
                      }}
                    """
