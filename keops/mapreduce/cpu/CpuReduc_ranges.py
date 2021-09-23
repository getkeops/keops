from keops.binders.cpp.Cpu_link_compile import Cpu_link_compile
from keops.binders.binders_definitions import binders_definitions
from keops.broadcast_batch_dimensions import (
    define_fill_shapes_function,
    define_broadcast_index_function,
    define_vect_broadcast_index_function,
)
from keops.mapreduce.cpu.CpuAssignZero import CpuAssignZero
from keops.mapreduce.MapReduce import MapReduce
from keops.utils.code_gen_utils import (
    c_variable,
    c_array,
    c_include,
)
from keops.config.config import use_OpenMP


class CpuReduc_ranges(MapReduce, Cpu_link_compile):
    # class for generating the final C++ code, Cpu version

    AssignZero = CpuAssignZero

    def __init__(self, *args):
        MapReduce.__init__(self, *args)
        Cpu_link_compile.__init__(self)
        self.dimy = self.varloader.dimy

    def get_code(self):

        super().get_code()

        i = self.i
        j = self.j
        dtype = self.dtype
        red_formula = self.red_formula
        fout = self.fout
        outi = self.outi
        acc = self.acc
        acctmp = self.acctmp
        arg = self.arg
        args = self.args
        nargs = len(args)

        xi = self.xi
        yj = c_array(dtype, self.varloader.dimy, "yj")
        param_loc = self.param_loc

        varloader = self.varloader
        table = varloader.table(xi, yj, param_loc)

        nvarsi, nvarsj, nvarsp = (
            len(self.varloader.Varsi),
            len(self.varloader.Varsj),
            len(self.varloader.Varsp),
        )

        tagHostDevice, tagCpuGpu, tag1D2D = (
            self.tagHostDevice,
            self.tagCpuGpu,
            self.tag1D2D,
        )

        sum_scheme = self.sum_scheme

        indices_i = c_array("int", nvarsi, "indices_i")
        indices_j = c_array("int", nvarsj, "indices_j")
        indices_p = c_array("int", nvarsp, "indices_p")
        imstartx = c_variable("int", "i-start_x")
        jmstarty = c_variable("int", "j-start_y")

        self.headers += c_include("cmath", "stdlib.h")
        if use_OpenMP:
            self.headers += c_include("omp.h")

        self.code = f"""
                        {self.headers}
                        #define __INDEX__ int
                  
                        {binders_definitions(dtype, red_formula, varloader)}
                        #include "Sizes_no_template.h"
                        #include "Ranges_no_template.h"
                        
                        {define_fill_shapes_function(red_formula)}
                        {define_broadcast_index_function()}
                        {define_vect_broadcast_index_function()}
                        
                        int CpuConv_ranges(int nx, int ny, 
                                            int nbatchdims, int* shapes,
                                            int nranges_x, int nranges_y, __INDEX__** ranges,
                                            {dtype}* out, {dtype} **{arg.id}) {{
                                                
                            // Separate and store the shapes of the "i" and "j" variables + parameters --------------
                            //
                            // shapes is an array of size (1+nargs)*(nbatchdims+3), which looks like:
                            // [ A, .., B, M, N, D_out]  -> output
                            // [ A, .., B, M, 1, D_1  ]  -> "i" variable
                            // [ A, .., B, 1, N, D_2  ]  -> "j" variable
                            // [ A, .., B, 1, 1, D_3  ]  -> "parameter"
                            // [ A, .., 1, M, 1, D_4  ]  -> N.B.: we support broadcasting on the batch dimensions!
                            // [ 1, .., 1, M, 1, D_5  ]  ->      (we'll just ask users to fill in the shapes with *explicit* ones)
    
                            int shapes_i[({nvarsi}) * (nbatchdims + 1)], shapes_j[{nvarsj} * (nbatchdims + 1)],
                                    shapes_p[{nvarsp} * (nbatchdims + 1)];
                            
                            // First, we fill shapes_i with the "relevant" shapes of the "i" variables,
                            // making it look like, say:
                            // [ A, .., B, M]
                            // [ A, .., 1, M]
                            // [ A, .., A, M]
                            // Then, we do the same for shapes_j, but with "N" instead of "M".
                            // And finally for the parameters, with "1" instead of "M".
                            fill_shapes(nbatchdims, shapes, shapes_i, shapes_j, shapes_p);
                            
                            // Actual for-for loop -----------------------------------------------------

                            {param_loc.declare()}
                            {varloader.load_vars("p", param_loc, args)}  // If nbatchdims == 0, the parameters are fixed once and for all
                            
                            // Set the output to zero, as the ranges may not cover the full output -----
                            {acctmp.declare()} // __TYPEACC__ acctmp[DIMRED];
                            for (int i = 0; i < nx; i++) {{
                                {red_formula.InitializeReduction(acctmp)}
                                {red_formula.FinalizeOutput(acctmp, outi, i)}
                            }}
                            
                            
                            // N.B.: In the following code, we assume that the x-ranges do not overlap.
                            //       Otherwise, we'd have to assume that DIMRED == DIMOUT
                            //       or allocate a buffer of size nx * DIMRED. This may be done in the future.
                            // Cf. reduction.h: 
                            //    FUN::tagJ = 1 for a reduction over j, result indexed by i
                            //    FUN::tagJ = 0 for a reduction over i, result indexed by j
    
                            int nranges = {red_formula.tagJ} ? nranges_x : nranges_y;
                            __INDEX__* ranges_x = {red_formula.tagJ} ? ranges[0] : ranges[3];
                            __INDEX__* slices_x = {red_formula.tagJ} ? ranges[1] : ranges[4];
                            __INDEX__* ranges_y = {red_formula.tagJ} ? ranges[2] : ranges[5];

                            int indices_i[{nvarsi}], indices_j[{nvarsj}], indices_p[{nvarsp}];  // Buffers for the "broadcasted indices"
                            for (int k = 0; k < {nvarsi}; k++) {{ indices_i[k] = 0; }}  // Fill the "offsets" with zeroes,
                            for (int k = 0; k < {nvarsj}; k++) {{ indices_j[k] = 0; }}  // the default value when nbatchdims == 0.
                            for (int k = 0; k < {nvarsp}; k++) {{ indices_p[k] = 0; }}
                            
                            
                            for (int range_index = 0; range_index < nranges; range_index++) {{
                                __INDEX__ start_x = ranges_x[2 * range_index];
                                __INDEX__ end_x = ranges_x[2 * range_index + 1];
                                __INDEX__ start_slice = (range_index < 1) ? 0 : slices_x[range_index - 1];
                                __INDEX__ end_slice = slices_x[range_index];

                                // If needed, compute the "true" start indices of the range, turning
                                // the "abstract" index start_x into an array of actual "pointers/offsets" stored in indices_i:
                                if (nbatchdims > 0) {{
                                    vect_broadcast_index(start_x, nbatchdims, {nvarsi}, shapes, shapes_i, indices_i);
                                    // And for the parameters, too:
                                    vect_broadcast_index(range_index, nbatchdims, {nvarsp}, shapes, shapes_p, indices_p);
                                    {varloader.load_vars("p", param_loc, args, offsets=indices_p)}  // Load the paramaters, once per tile
                                }}
                            
                                #pragma omp parallel for   
                                for (__INDEX__ i = start_x; i < end_x; i++) {{
                                    {xi.declare()}
                                    {yj.declare()}
                                    {fout.declare()}
                                    {acc.declare()}
                                    {sum_scheme.declare_temporary_accumulator()}
                                    if (nbatchdims == 0) {{
                                        {varloader.load_vars("i", xi, args, row_index=i)}
                                    }} else {{
                                        {varloader.load_vars("i", xi, args, row_index=imstartx, offsets=indices_i)}
                                    }}
                                    {red_formula.InitializeReduction(acc)}
                                    {sum_scheme.initialize_temporary_accumulator()}
                                    for (__INDEX__ slice = start_slice; slice < end_slice; slice++) {{
                                        __INDEX__ start_y = ranges_y[2 * slice];
                                        __INDEX__ end_y = ranges_y[2 * slice + 1];
      
                                        // If needed, compute the "true" start indices of the range, turning
                                        // the "abstract" index start_y into an array of actual "pointers/offsets" stored in indices_j:
                                        if (nbatchdims > 0) {{
                                            vect_broadcast_index(start_y, nbatchdims, {nvarsj}, shapes, shapes_j, indices_j);
                                        }}
                                        if (nbatchdims == 0) {{
                                            for (int j = start_y; j < end_y; j++) {{
                                                {varloader.load_vars("j", yj, args, row_index=j)}
                                                {red_formula.formula(fout,table)}
                                                {sum_scheme.accumulate_result(acc, fout, j)}
                                            }}
                                        }} else {{
                                            for (int j = start_y; j < end_y; j++) {{
                                                {varloader.load_vars("j", yj, args, row_index=jmstarty, offsets=indices_j)}
                                                {red_formula.formula(fout,table)}
                                                {sum_scheme.accumulate_result(acc, fout, jmstarty)}
                                            }}
                                        }}
                                    }}
                                    {sum_scheme.final_operation(acc)}
                                    {red_formula.FinalizeOutput(acc, outi, i)}
                                }}
                            }}
                            return 0;
                        }}
                    """

        self.code += f"""    
                    
                    #include "stdarg.h"
                    
                    extern "C" int launch_keops_{dtype}(const char* target_file_name, int tagHostDevice, int dimY, int nx, int ny, 
                                                        int device_id, int tagI, int tagZero, int use_half, 
                                                        int tag1D2D, int dimred,
                                                        int cuda_block_size, int use_chunk_mode,
                                                        int *indsi, int *indsj, int *indsp, 
                                                        int dimout, 
                                                        int *dimsx, int *dimsy, int *dimsp,
                                                        int **ranges, int *shapeout, {dtype}* out, int nargs, ...) {{
                        
                        // reading arguments
                        va_list ap;
                        va_start(ap, nargs);
                        {dtype} *arg[nargs];
                        for (int i=0; i<nargs; i++)
                            arg[i] = va_arg(ap, {dtype}*);
                        int *argshape[nargs];
                        for (int i=0; i<nargs; i++)
                            argshape[i] = va_arg(ap, int*);
                        va_end(ap);
                        
                        Sizes SS(nargs, arg, argshape, nx, ny);
                        
                        /* To be used with the Size.h (templated version)
                        Sizes SS(nargs, arg, argshape, nx, ny,
                                 tagI, use_half, dimout,
                                 indsi, indsj, indsp,
                                 dimsx, dimsy, dimsp);
                        */
                        
                        if (use_half)
                          SS.switch_to_half2_indexing();

                        Ranges RR(SS, ranges);
                        
                        nx = SS.nx;
                        ny = SS.ny;
                        
                        if (tagI==1) {{
                            int tmp = ny;
                            ny = nx;
                            nx = tmp;
                        }}
                        
                        return CpuConv_ranges(nx, ny, SS.nbatchdims, SS.shapes,
                                                RR.nranges_x, RR.nranges_y, RR.castedranges,
                                                out, arg);
                    }}
                """
