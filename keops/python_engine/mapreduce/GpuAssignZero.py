from keops.python_engine.mapreduce.MapReduce import MapReduce
from keops.python_engine.code_gen_utils import c_include, signature_list, c_zero_float, call_list
from keops.python_engine.link_compile import Gpu_link_compile


class GpuAssignZero(MapReduce, Gpu_link_compile):
    # class for generating the final C++ code, Gpu version

    def __init__(self, *args):
        MapReduce.__init__(self, *args)
        Gpu_link_compile.__init__(self)

    def get_code(self):

        super().get_code()

        outi = self.outi
        dtype = self.dtype
        args = self.args
        varloader = self.varloader

        if dtype == "half2":
            self.headers += c_include("cuda_fp16.h")

        self.code = f"""
                        {self.headers}

                        __global__ void AssignZeroGpu(int nx, int ny, {dtype} *out, {signature_list(args)}) {{
    
                          // get the index of the current thread
                          int i = blockIdx.x * blockDim.x + threadIdx.x;

                          if (i < nx) {{
                            {outi.assign(c_zero_float)}
                          }}

                        }}



                        extern "C" __host__ int launch_keops(int nx, int ny, int device_id, int *ranges, {dtype} *out, {signature_list(args)}) {{

                            // device_id is provided, so we set the GPU device accordingly
                            // Warning : is has to be consistent with location of data
                            cudaSetDevice(device_id);
	
                            // Compute on device : grid and block are both 1d

                            //SetGpuProps(device_id);

                            dim3 blockSize;

                            blockSize.x = 32;
	
                            dim3 gridSize;
                            gridSize.x = nx / blockSize.x + (nx % blockSize.x == 0 ? 0 : 1);

                            AssignZeroGpu <<< gridSize, blockSize, blockSize.x * {varloader.dimy} * sizeof({dtype}) >>> (nx, ny, out, {call_list(args)});
    
                            // block until the device has completed
                            cudaDeviceSynchronize();

                            //CudaCheckError();

                            return 0;
                        }}
                    """