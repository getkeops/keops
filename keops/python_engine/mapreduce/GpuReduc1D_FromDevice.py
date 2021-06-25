from keops.python_engine.mapreduce.GpuReduc1D import GpuReduc1D



class GpuReduc1D_FromDevice(GpuReduc1D):

    def get_code(self, for_jit=False):

        super().get_code(for_jit=for_jit)

        if not for_jit:

            self.code += f"""
            
                        #include "stdarg.h"
                        
                        extern "C" __host__ int launch_keops(const char* ptx_file_name, int dimY, int nx, int ny, int device_id, int tagI, 
                                                            int **ranges, {dtype} *out, int nargs, ...) {{
                            
                            if (tagI==1) {{
                                int tmp = ny;
                                ny = nx;
                                nx = tmp;
                            }}
                            
                            // reading arguments
                            va_list ap;
                            va_start(ap, nargs);
                            {dtype} *arg[nargs];
                            for (int i=0; i<nargs; i++)
                                arg[i] = va_arg(ap, {dtype}*);
                            va_end(ap);
    
                            // device_id is provided, so we set the GPU device accordingly
                            // Warning : is has to be consistent with location of data
                            cudaSetDevice(device_id);

                            // Compute on device : grid and block are both 1d

                            //SetGpuProps(device_id);

                            dim3 blockSize;

                            blockSize.x = 32;

                            dim3 gridSize;
                            gridSize.x = nx / blockSize.x + (nx % blockSize.x == 0 ? 0 : 1);

                            GpuConv1DOnDevice <<< gridSize, blockSize, blockSize.x * dimY * sizeof({dtype}) >>> (nx, ny, out, arg);
    
                            // block until the device has completed
                            cudaDeviceSynchronize();

                            //CudaCheckError();

                            return 0;
                        }}
                    """
