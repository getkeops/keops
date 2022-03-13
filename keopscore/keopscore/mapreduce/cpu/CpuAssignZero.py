from keopscore import debug_ops_at_exec
from keopscore.binders.cpp.Cpu_link_compile import Cpu_link_compile
from keopscore.mapreduce.MapReduce import MapReduce
from keopscore.utils.code_gen_utils import (
    c_include,
    c_zero_float,
)
import keopscore


class CpuAssignZero(MapReduce, Cpu_link_compile):
    # class for generating the final C++ code, Cpu version

    def __init__(self, *args):
        MapReduce.__init__(self, *args)
        Cpu_link_compile.__init__(self)
        self.dimy = self.varloader.dimy

    def get_code(self):

        super().get_code()

        outi = self.outi
        dtype = self.dtype
        arg = self.arg
        args = self.args

        headers = ["stdlib.h"]
        if keopscore.config.config.use_OpenMP:
            headers.append("omp.h")
        if debug_ops_at_exec:
            headers.append("iostream")
        self.headers += c_include(*headers)

        self.code = f"""
{self.headers}

template < typename TYPE >
int AssignZeroCpu_{self.gencode_filename}(int nx, int ny, TYPE* out,  TYPE **{arg.id}) {{
    #pragma omp parallel for
    for (int i = 0; i < nx; i++) {{
        {outi.assign(c_zero_float)}
    }}
    return 0;
}}

#include "stdarg.h"
#include <vector>

template < typename TYPE >
int launch_keops_{self.gencode_filename}(int nx, int ny, int tagI, TYPE *out, TYPE **arg) {{

    if (tagI==1) {{
        int tmp = ny;
        ny = nx;
        nx = tmp;
    }}

    return AssignZeroCpu_{self.gencode_filename}< TYPE > (nx, ny, out, arg);

}}

template < typename TYPE >
int launch_keops_cpu_{self.gencode_filename}(int dimY,
                                         int nx,
                                         int ny,
                                         int tagI,
                                         int tagZero,
                                         int use_half,
                                         int dimred,
                                         int use_chunk_mode,
                                         std::vector< int > indsi, std::vector< int > indsj, std::vector< int > indsp,
                                         int dimout,
                                         std::vector< int > dimsx, std::vector< int > dimsy, std::vector< int > dimsp,
                                         int **ranges,
                                         std::vector< int > shapeout, TYPE *out,
                                         TYPE **arg,
                                         std::vector< std::vector< int > > argshape) {{


    return launch_keops_{self.gencode_filename}< TYPE > (nx, ny, tagI, out, arg);

}}

                    """
