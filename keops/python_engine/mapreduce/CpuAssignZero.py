from keops.python_engine.mapreduce.MapReduce import MapReduce
from keops.python_engine.utils.code_gen_utils import (
    c_include,
    signature_list,
    c_zero_float,
    call_list,
)
from keops.python_engine.compilation import Cpu_link_compile


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

        self.headers += c_include("omp.h")

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
                        
                        extern "C" int launch_keops_{dtype}(const char* ptx_file_name, int tagHostDevice, int dimY, int nx, int ny, 
                                                            int device_id, int tagI, int tagZero, int use_half, 
                                                            int tag1D2D,
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
                            
                            if (tagI==1) {{
                                int tmp = ny;
                                ny = nx;
                                nx = tmp;
                            }}
                            
                            return AssignZeroCpu(nx, ny, out, arg);
                        }}
                    """
