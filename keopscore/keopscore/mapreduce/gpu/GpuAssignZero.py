from keopscore.binders.nvrtc.Gpu_link_compile import Gpu_link_compile
from keopscore.mapreduce.MapReduce import MapReduce
from keopscore.utils.code_gen_utils import (
    c_include,
    c_zero_float,
)


class GpuAssignZero(MapReduce, Gpu_link_compile):
    # class for generating the final C++ code, Gpu version

    def __init__(self, *args):
        MapReduce.__init__(self, *args)
        Gpu_link_compile.__init__(self)
        self.dimy = self.varloader.dimy

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
