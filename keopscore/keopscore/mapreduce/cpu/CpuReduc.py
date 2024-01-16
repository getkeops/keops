import keopscore
from keopscore.binders.cpp.Cpu_link_compile import Cpu_link_compile
from keopscore.mapreduce.cpu.CpuAssignZero import CpuAssignZero
from keopscore.mapreduce.MapReduce import MapReduce
from keopscore.utils.code_gen_utils import c_include
import keopscore


class CpuReduc(MapReduce, Cpu_link_compile):
    """
    class for generating the final C++ code, Cpu version
    """

    AssignZero = CpuAssignZero

    def __init__(self, *args):
        MapReduce.__init__(self, *args)
        Cpu_link_compile.__init__(self)
        self.dimy = self.varloader.dimy

    def get_code(self):
        super().get_code()

        i = self.i
        j = self.j
        red_formula = self.red_formula
        fout = self.fout
        outi = self.outi
        acc = self.acc
        arg = self.arg
        args = self.args
        table = self.varloader.direct_table(args, i, j)
        sum_scheme = self.sum_scheme

        headers = ["cmath", "stdlib.h"]
        if keopscore.config.config.use_OpenMP:
            headers.append("omp.h")
        if keopscore.debug_ops_at_exec:
            headers.append("iostream")
        self.headers += c_include(*headers)

        self.code = f"""
{self.headers}
template < typename TYPE > 
int CpuConv_{self.gencode_filename}(signed long int nx, signed long int ny, TYPE* out, TYPE **{arg.id}) {{
    #pragma omp parallel for
    for (signed long int i = 0; i < nx; i++) {{
        {fout.declare()}
        {acc.declare()}
        {sum_scheme.declare_temporary_accumulator()}
        {red_formula.InitializeReduction(acc)}
        {sum_scheme.initialize_temporary_accumulator()}
        for (signed long int j = 0; j < ny; j++) {{
            {red_formula.formula(fout,table)}
            {sum_scheme.accumulate_result(acc, fout, j)}
            {sum_scheme.periodic_accumulate_temporary(acc, j)}
        }}
        {sum_scheme.final_operation(acc)}
        {red_formula.FinalizeOutput(acc, outi, i)}
    }}
    return 0;
}}
                    """

        self.code += f"""
#include "stdarg.h"
#include <vector>

template < typename TYPE > 
int launch_keops_{self.gencode_filename}(signed long int nx, signed long int ny, int tagI, TYPE *out, TYPE **arg) {{
    
    if (tagI==1) {{
        signed long int tmp = ny;
        ny = nx;
        nx = tmp;
    }}
    
    return CpuConv_{self.gencode_filename}< TYPE >(nx, ny, out, arg);

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

    
    return launch_keops_{self.gencode_filename} < TYPE >(nx, ny, tagI, out, arg);

}}
                """
