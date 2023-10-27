import keopscore
from keopscore.binders.cpp.Cpu_link_compile import Cpu_link_compile

from keopscore.mapreduce.cpu.CpuAssignZero import CpuAssignZero
from keopscore.mapreduce.MapReduce import MapReduce
from keopscore.utils.code_gen_utils import (
    c_variable,
    c_array,
    c_include,
)
import keopscore


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

        headers = ["cmath", "stdlib.h"]
        if keopscore.config.config.use_OpenMP:
            headers.append("omp.h")
        if keopscore.debug_ops_at_exec:
            headers.append("iostream")
        self.headers += c_include(*headers)

        self.code = f"""
{self.headers}

#define do_keops_checks 0
#if do_keops_checks
    #include <string>
    #include <iostream>
#endif

#if do_keops_checks
    #define Error_msg_no_cuda "[KeOps] This KeOps shared object has been compiled without cuda support: - 1) to perform computations on CPU, simply set tagHostDevice to 0 - 2) to perform computations on GPU, please recompile the formula with a working version of cuda."
    void keops_error(std::string message) {{
    	throw std::runtime_error(message);
    }}
#endif

#if do_keops_checks
void check_nargs(int nargs, int nminargs) {{
    if(nargs<nminargs) {{
        keops_error("[KeOps] : not enough input arguments");
    }}
}}
#endif

#include "include/Sizes.h"
#include "include/ranges_utils.h"
#include "include/Ranges.h"

template< typename TYPE>                 
int CpuConv_ranges_{self.gencode_filename}(signed long int nx, signed long int ny, 
                    int nbatchdims, signed long int* shapes,
                    std::vector< int > indsi, std::vector< int > indsj, std::vector< int > indsp,
                    signed long int nranges_x, signed long int nranges_y, signed long int **ranges,
                    TYPE* out, std::vector< signed long int > shapeout, TYPE **{arg.id}) {{
                        
    int sizei = indsi.size();
    int sizej = indsj.size();
    int sizep = indsp.size();
    
    // Separate and store the shapes of the "i" and "j" variables + parameters --------------
    //
    // shapes is an array of size (1+nargs)*(nbatchdims+3), which looks like:
    // [ A, .., B, M, N, D_out]  -> output
    // [ A, .., B, M, 1, D_1  ]  -> "i" variable
    // [ A, .., B, 1, N, D_2  ]  -> "j" variable
    // [ A, .., B, 1, 1, D_3  ]  -> "parameter"
    // [ A, .., 1, M, 1, D_4  ]  -> N.B.: we support broadcasting on the batch dimensions!
    // [ 1, .., 1, M, 1, D_5  ]  ->      (we'll just ask users to fill in the shapes with *explicit* ones)
            
    signed long int shapes_i[sizei * (nbatchdims + 1)], shapes_j[sizej * (nbatchdims + 1)], shapes_p[sizep * (nbatchdims + 1)];

    
    // First, we fill shapes_i with the "relevant" shapes of the "i" variables,
    // making it look like, say:
    // [ A, .., B, M]
    // [ A, .., 1, M]
    // [ A, .., A, M]
    // Then, we do the same for shapes_j, but with "N" instead of "M".
    // And finally for the parameters, with "1" instead of "M".
    fill_shapes(nbatchdims, shapes, shapes_i, shapes_j, shapes_p,  {red_formula.tagJ}, indsi, indsj, indsp);
    
    // Actual for-for loop -----------------------------------------------------

    {param_loc.declare()}
    {varloader.load_vars("p", param_loc, args)}  // If nbatchdims == 0, the parameters are fixed once and for all
    
    // Set the output to zero, as the ranges may not cover the full output -----
    {acctmp.declare()} // __TYPEACC__ acctmp[DIMRED];

    // for some reason the nx value is not correct for very special cases (like Zero reduction..)
    // so we compute the true value from the input shapeout...
    signed long int true_nx = 1;
    for (signed long int k=0; k<shapeout.size()-1; k++) {{
        true_nx *= shapeout[k];
    }}

    for (signed long int i = 0; i < true_nx; i++) {{
        {red_formula.InitializeReduction(acctmp)}
        {red_formula.FinalizeOutput(acctmp, outi, i)}
    }}
    
    // N.B.: In the following code, we assume that the x-ranges do not overlap.
    //       Otherwise, we'd have to assume that DIMRED == DIMOUT
    //       or allocate a buffer of size nx * DIMRED. This may be done in the future.
    // Cf. reduction.h: 
    //    FUN::tagJ = 1 for a reduction over j, result indexed by i
    //    FUN::tagJ = 0 for a reduction over i, result indexed by j

    signed long int nranges = {red_formula.tagJ} ? nranges_x : nranges_y;
    signed long int* ranges_x = {red_formula.tagJ} ? ranges[0] : ranges[3];
    signed long int* slices_x = {red_formula.tagJ} ? ranges[1] : ranges[4];
    signed long int* ranges_y = {red_formula.tagJ} ? ranges[2] : ranges[5];

    signed long int indices_i[sizei], indices_j[sizej], indices_p[sizep];  // Buffers for the "broadcasted indices"
    for (signed long int k = 0; k < sizei; k++) {{ indices_i[k] = 0; }}  // Fill the "offsets" with zeroes,
    for (signed long int k = 0; k < sizej; k++) {{ indices_j[k] = 0; }}  // the default value when nbatchdims == 0.
    for (signed long int k = 0; k < sizep; k++) {{ indices_p[k] = 0; }}
    
    
    for (signed long int range_index = 0; range_index < nranges; range_index++) {{
        signed long int start_x = ranges_x[2 * range_index];
        signed long int end_x = ranges_x[2 * range_index + 1];
        signed long int start_slice = (range_index < 1) ? 0 : slices_x[range_index - 1];
        signed long int end_slice = slices_x[range_index];

        // If needed, compute the "true" start indices of the range, turning
        // the "abstract" index start_x into an array of actual "pointers/offsets" stored in indices_i:
        if (nbatchdims > 0) {{
            vect_broadcast_index(start_x, nbatchdims, sizei, shapes, shapes_i, indices_i);
            // And for the parameters, too:
            vect_broadcast_index(range_index, nbatchdims, sizep, shapes, shapes_p, indices_p);
            {varloader.load_vars("p", param_loc, args, offsets=indices_p)}  // Load the paramaters, once per tile
        }}
    
        #pragma omp parallel for   
        for (signed long int i = start_x; i < end_x; i++) {{
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
            for (signed long int slice = start_slice; slice < end_slice; slice++) {{
                 signed long int start_y = ranges_y[2 * slice];
                 signed long int end_y = ranges_y[2 * slice + 1];

                // If needed, compute the "true" start indices of the range, turning
                // the "abstract" index start_y into an array of actual "pointers/offsets" stored in indices_j:
                if (nbatchdims > 0) {{
                    vect_broadcast_index(start_y, nbatchdims, sizej, shapes, shapes_j, indices_j);
                }}
                if (nbatchdims == 0) {{
                    for (signed long int j = start_y; j < end_y; j++) {{
                        {varloader.load_vars("j", yj, args, row_index=j)}
                        {red_formula.formula(fout,table)}
                        {sum_scheme.accumulate_result(acc, fout, j)}
                    }}
                }} else {{
                    for (signed long int j = start_y; j < end_y; j++) {{
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
#include <vector>

template < typename TYPE >
int launch_keops_{self.gencode_filename}(signed long int nx, signed long int ny, 
                                         int tagI, int use_half, signed long int dimred,
                                         int use_chunk_mode,
                                         std::vector< int > indsi, std::vector< int > indsj, std::vector< int > indsp,
                                         signed long int dimout,
                                         std::vector< signed long int > dimsx, std::vector< signed long int > dimsy, std::vector< signed long int > dimsp,
                                         signed long int **ranges, 
                                         TYPE *out, std::vector< signed long int > shapeout, int nargs, TYPE** arg,
                                         std::vector<std::vector< signed long int >> argshape) {{
    
    Sizes< TYPE > SS (nargs, arg, argshape, nx, ny,tagI, use_half,
                      dimout,
                      indsi, indsj, indsp,
                      dimsx, dimsy, dimsp
                      );
    
    if (use_half)
      SS.switch_to_half2_indexing();

    Ranges < TYPE > RR(SS, ranges);
    
    nx = SS.nx;
    ny = SS.ny;
    
    if (tagI==1) {{
        signed long int tmp = ny;
        ny = nx;
        nx = tmp;
    
        std::vector< int > tmp_v_ind;
        
        tmp_v_ind = indsj;
        indsj = indsi;
        indsi = tmp_v_ind;

        std::vector< signed long int > tmp_v_dim;

        tmp_v_dim = dimsy;
        dimsy = dimsx;
        dimsx = tmp_v_dim;
    }}
    
    return CpuConv_ranges_{self.gencode_filename}< TYPE> (nx, ny, SS.nbatchdims, SS.shapes,
                                                          indsi, indsj, indsp,
                                                          RR.nranges_x, RR.nranges_y, RR.castedranges,
                                                          out, shapeout, arg);
}}

template < typename TYPE >
int launch_keops_cpu_{self.gencode_filename}(signed long int dimY,
                                             signed long int nx,
                                             signed long int ny,
                                             int tagI,
                                             int tagZero,
                                             int use_half,
                                             signed long int dimred,
                                             int use_chunk_mode,
                                             std::vector< int > indsi, std::vector< int > indsj, std::vector< int > indsp,
                                             signed long int dimout,
                                             std::vector< signed long int > dimsx, std::vector< signed long int > dimsy, std::vector< signed long int > dimsp,
                                             signed long int **ranges,
                                             std::vector< signed long int > shapeout, TYPE *out,
                                             TYPE **arg,
                                             std::vector< std::vector< signed long int > > argshape) {{
    

    
    return launch_keops_{self.gencode_filename}< TYPE >(nx,
                                                        ny,
                                                        tagI,
                                                        use_half,
                                                        dimred,
                                                        use_chunk_mode,
                                                        indsi, indsj, indsp,
                                                        dimout,
                                                        dimsx, dimsy, dimsp,
                                                        ranges,
                                                        out, 
                                                        shapeout,
                                                        argshape.size(), 
                                                        arg, 
                                                        argshape);
}}
                        
                """
