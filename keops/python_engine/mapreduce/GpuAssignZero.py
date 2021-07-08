from keops.python_engine.mapreduce.MapReduce import MapReduce
from keops.python_engine.utils.code_gen_utils import (
    c_include,
    signature_list,
    c_zero_float,
    call_list,
)
from keops.python_engine.compilation import Gpu_link_compile


class GpuAssignZero(MapReduce, Gpu_link_compile):
    # class for generating the final C++ code, Gpu version

    def __init__(self, *args):
        MapReduce.__init__(self, *args)
        Gpu_link_compile.__init__(self)

    def get_code(self):

        super().get_code()

        outi = self.outi
        dtype = self.dtype
        arg = self.arg
        varloader = self.varloader

        if dtype == "half2":
            self.headers += c_include("cuda_fp16.h")

        self.code = f"""
                        {self.headers}

                        extern "C" __global__ void GpuConv1DOnDevice(int nx, int ny, {dtype} *out, {dtype} **{arg.id}) {{
    
                          // get the index of the current thread
                          int i = blockIdx.x * blockDim.x + threadIdx.x;

                          if (i < nx) {{
                            {outi.assign(c_zero_float)}
                          }}

                        }}
                    """
