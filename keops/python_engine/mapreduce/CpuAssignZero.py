from keops.python_engine.mapreduce.MapReduce import MapReduce
from keops.python_engine.utils.code_gen_utils import c_include, signature_list, c_zero_float, call_list
from keops.python_engine.link_compile import Cpu_link_compile


class CpuAssignZero(MapReduce, Cpu_link_compile):
    # class for generating the final C++ code, Cpu version

    def __init__(self, *args):
        MapReduce.__init__(self, *args)
        Cpu_link_compile.__init__(self)

    def get_code(self):

        super().get_code()

        outi = self.outi
        dtype = self.dtype
        args = self.args

        self.headers += c_include("omp.h")

        self.code = f"""
                        {self.headers}

                        extern "C" int AssignZeroCpu(int nx, int ny, {dtype}* out, {signature_list(args)}) {{
                            #pragma omp parallel for
                            for (int i = 0; i < nx; i++) {{
                                {outi.assign(c_zero_float)}
                            }}
                            return 0;
                        }}
                        
                        extern "C" int launch_keops(int nx, int ny, int device_id, int *ranges, {dtype}* out, {signature_list(args)}) {{
                            return AssignZeroCpu(nx, ny, out, {call_list(args)});
                        }}
                    """