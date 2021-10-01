template <typename TYPE>
class KeOps_module {
    public:
        CUdevice cuDevice;
        CUcontext ctx;
        CUmodule module;
        char *target;
        CUdeviceptr buffer;
        int nargs;
        void SetContext() ;
        void Read_Target(const char *target_file_name);
        KeOps_module(int device_id, int nargs_, const char *target_file_name);
        ~KeOps_module();
        int operator () (int tagHostDevice, int dimY, int nx, int ny,
                         int tagI, int tagZero, int use_half,
                         int tag1D2D, int dimred,
                         int cuda_block_size, int use_chunk_mode,
                         int *indsi, int *indsj, int *indsp,
                         int dimout,
                         int *dimsx, int *dimsy, int *dimsp,
                         const std::vector<void*>& ranges_v,
                         int *shapeout, void *out_void,
                         const std::vector<void*>& arg_v,
                         const std::vector<int*>& argshape_v
                         );
};
