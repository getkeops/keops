from keops import debug_ops
from keops.binders.cpp.Cpu_link_compile import Cpu_link_compile
from keops.mapreduce.MapReduce import MapReduce
from keops.utils.code_gen_utils import (
    c_include,
    c_zero_float,
)
from keops.config.config import use_OpenMP


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
        if use_OpenMP:
            headers.append("omp.h")
        if debug_ops:
            headers.append("iostream")
        self.headers += c_include(*headers)

        self.code = f"""
                        {self.headers}

                        extern "C" int AssignZeroCpu(int nx, int ny, {dtype}* out, {dtype} **{arg.id}) {{
                            #pragma omp parallel for
                            for (int i = 0; i < nx; i++) {{
                                {outi.assign(c_zero_float)}
                            }}
                            return 0;
                        }}
                        
                        #include "stdarg.h"
                        #include <vector>
                        
                        int launch_keops(int nx, int ny, int tagI, {dtype} *out, {dtype} **arg) {{
                        
                            if (tagI==1) {{
                                int tmp = ny;
                                ny = nx;
                                nx = tmp;
                            }}
                        
                            return AssignZeroCpu(nx, ny, out, arg);

                        }}
                        
                        int launch_keops_cpu_{self.gencode_filename}(int nx, int ny, int tagI, int use_half, 
                                                 const std::vector<void*>& ranges_v,
                                                 void *out_void, int nargs, 
                                                 const std::vector<void*>& arg_v,
                                                 const std::vector<int*>& argshape_v) {{
                        
                            {dtype} **arg = ({dtype}**) arg_v.data();
                            {dtype} *out = ({dtype}*) out_void;
                        
                            return launch_keops(nx, ny, tagI, out, arg);

                        }}
                        
                        extern "C" int launch_keops_{dtype}(const char* target_file_name, int tagHostDevice, int dimY, int nx, int ny, 
                                                            int device_id, int tagI, int tagZero, int use_half, 
                                                            int tag1D2D, int dimred,
                                                            int cuda_block_size, int use_chunk_mode,
                                                            int *indsi, int *indsj, int *indsp, 
                                                            int dimout, 
                                                            int *dimsx, int *dimsy, int *dimsp, 
                                                            int **ranges, int *shapeout, {dtype} *out, int nargs, ...) {{
                            
                            // reading arguments
                            va_list ap;
                            va_start(ap, nargs);
                            {dtype} *arg[nargs];
                            for (int i=0; i<nargs; i++)
                                arg[i] = va_arg(ap, {dtype}*);
                            va_end(ap);
                            
                            return launch_keops(nx, ny, tagI, out, arg);
                        }}
                    """
