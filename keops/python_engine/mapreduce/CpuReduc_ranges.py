from keops.python_engine.mapreduce.MapReduce import MapReduce
from keops.python_engine.mapreduce.CpuAssignZero import CpuAssignZero
from keops.python_engine.utils.code_gen_utils import c_variable, c_array, c_include, signature_list, call_list, varseq_to_array
from keops.python_engine.compilation import Cpu_link_compile
from keops.python_engine import use_jit
from keops.python_engine.binders.binders_definitions import binders_definitions
from keops.python_engine.broadcast_batch_dimensions import define_fill_shapes_function, define_broadcast_index_function, define_vect_broadcast_index_function

class CpuReduc_ranges(MapReduce, Cpu_link_compile):
    # class for generating the final C++ code, Cpu version

    AssignZero = CpuAssignZero

    def __init__(self, *args):
        if use_jit:
            raise ValueError("JIT compiling not yet implemented in Cpu mode")
        MapReduce.__init__(self, *args)
        Cpu_link_compile.__init__(self)

    def get_code(self, for_jit=False):
        
        if for_jit:
            raise ValueError("JIT compiling not yet implemented in Cpu mode")

        super().get_code()

        i = self.i
        j = self.j
        dtype = self.dtype
        red_formula = self.red_formula
        fout = self.fout
        outi = self.outi
        acc = self.acc
        acctmp = self.acctmp
        args = self.args
        nargs = len(args)
        argshapes = self.argshapes
        
        xi = self.xi
        yj = c_array(dtype, self.varloader.dimy, "yj")
        param_loc = self.param_loc
        
        varloader = self.varloader
        table = varloader.table(xi, yj, param_loc)
        
        nvarsi, nvarsj, nvarsp = len(self.varloader.Varsi), len(self.varloader.Varsj), len(self.varloader.Varsp)
        
        tagHostDevice, tagCpuGpu, tag1D2D = self.tagHostDevice, self.tagCpuGpu, self.tag1D2D
        
        sum_scheme = self.sum_scheme
        
        indices_i = c_array("int", nvarsi, "indices_i")
        indices_j = c_array("int", nvarsj, "indices_j")
        indices_p = c_array("int", nvarsp, "indices_p")
        imstartx = c_variable("int", "i-start_x")
        jmstarty = c_variable("int", "j-start_y")

        self.headers += c_include("cmath", "omp.h")
        
        
        self.code = f"""
                        {self.headers}
                        #define __INDEX__ int32_t
                  
                        {binders_definitions(dtype, red_formula, varloader, tagHostDevice, tagCpuGpu, tag1D2D)}
                        #include "Sizes.h"
                        #include "Ranges.h"
                        
                        {define_fill_shapes_function(red_formula)}
                        {define_broadcast_index_function()}
                        {define_vect_broadcast_index_function()}
                        
                        int CpuConv_ranges(int nx, int ny, 
                                            int nbatchdims, int* shapes,
                                            int nranges_x, int nranges_y, __INDEX__** ranges,
                                            {dtype}* out, {signature_list(args)}) {{
                                                
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
                        
                        
                        extern "C" int launch_keops(int nx, int ny, int device_id, int *ranges, {dtype}* out, {signature_list(args)}, {signature_list(argshapes)}) {{
                            
                            {varseq_to_array(args, "args_ptr")}
                            {varseq_to_array(argshapes, "argshapes_ptr")}

                            Sizes SS({nargs}, args_ptr, argshapes_ptr, nx, ny);
                            
                            #if USE_HALF
                              SS.switch_to_half2_indexing();
                            #endif
                            
                            int nranges = 0;

                            Ranges RR(SS, nranges, NULL);  // N.B. third arg should be ranges
                            
                            return CpuConv_ranges(SS.nx, SS.ny, SS.nbatchdims, SS.shapes,
                                                      RR.nranges_x, RR.nranges_y, RR.castedranges,
                                                      out, {call_list(args)});
                        }}
                    """
