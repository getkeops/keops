from keops.python_engine.mapreduce.GpuReduc1D import GpuReduc1D



class GpuReduc1D_FromHost(GpuReduc1D):

    def get_code(self, for_jit=False):

        super().get_code(for_jit=for_jit)

        if not for_jit:
            
            nminargs = self.varloader.nminargs
            nargsi = self.varloader.nargsi
            nargsj = self.varloader.nargsj
            nargsp = self.varloader.nargsp
            dimx = self.varloader.dimx
            dimy = self.varloader.dimy
            dimp = self.varloader.dimp
            dimsx = self.varloader.dimsx
            dimsy = self.varloader.dimsy
            dimsp = self.varloader.dimsp
            indsi = self.varloader.indsi
            indsj = self.varloader.indsj
            indsp = self.varloader.indsp
            dimout = self.redformula.dim
            
            dtype = self.dtype
            

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
                            
                            
                            
                            
                            
                            // pointer to device output array
                            {dtype} *out_d;

                            // array of pointers to device input arrays
                            {dtype} **args_d;

                            void *p_data;
                            // single cudaMalloc
                            CudaSafeCall(cudaMalloc(&p_data,
                                                    sizeof({dtype} *) * {nminargs}
                                                        + sizeof({dtype}) * ({dimp} + nx * ({dimx} + {dimout}) + ny * {dimy})));

                            args_d = ({dtype} **) p_data;
                            {dtype} *dataloc = ({dtype} *) (args_d + {nminargs});
                            out_d = dataloc;
                            dataloc += nx*{dimout};

                            // host array of pointers to device data
                            {dtype} *ph[{nminargs}];

                              for (int k = 0; k < {nargsp}; k++) {{
                                int indk = {indsp[k]};
                                int nvals = {dimsp[k]};        
                                CudaSafeCall(cudaMemcpy(dataloc, args_h[indk], sizeof({dtype}) * nvals, cudaMemcpyHostToDevice));
                                ph[indk] = dataloc;
                                dataloc += nvals;
                              }}

                            for (int k = 0; k < {nargsi}; k++) {{
                              int indk = {indsi[k]};
                              int nvals = nx * {dimsx[k]};
                              CudaSafeCall(cudaMemcpy(dataloc, args_h[indk], sizeof({dtype}) * nvals, cudaMemcpyHostToDevice));
                              ph[indk] = dataloc;
                              dataloc += nvals;
                            }}

                              for (int k = 0; k < {nargsj}; k++) {{
                                int indk = {indsj[k]};
                                int nvals = ny * {dimsy[k]};
                                CudaSafeCall(cudaMemcpy(dataloc, args_h[indk], sizeof({dtype}) * nvals, cudaMemcpyHostToDevice));
                                ph[indk] = dataloc;
                                dataloc += nvals;
                              }}


                            // copy array of pointers
                            CudaSafeCall(cudaMemcpy(args_d, ph, {nminargs} * sizeof({dtype} *), cudaMemcpyHostToDevice));
                            
                            
                            
                            

                            // Compute on device : grid and block are both 1d

                            //SetGpuProps(device_id);

                            dim3 blockSize;

                            blockSize.x = 32;

                            dim3 gridSize;
                            gridSize.x = nx / blockSize.x + (nx % blockSize.x == 0 ? 0 : 1);

                            GpuConv1DOnDevice <<< gridSize, blockSize, blockSize.x * dimY * sizeof({dtype}) >>> (nx, ny, out_d, arg_d);
    
                            // block until the device has completed
                            cudaDeviceSynchronize();

                            //CudaCheckError();
                            
                            
                            
                            // Send data from device to host.
                            CudaSafeCall(cudaMemcpy(out, out_d, sizeof({dtype}) * (nx * {dimout}), cudaMemcpyDeviceToHost));
                            
                            // Free memory.
                            CudaSafeCall(cudaFree(p_data));

                            return 0;
                        }}
                    """
