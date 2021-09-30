template <typename TYPE>
class context {
public:
int current_device_id;
CUcontext ctx;
CUmodule module;
char *target;
CUdeviceptr buffer;
void SetDevice(int device_id);
void Read_Target(const char *target_file_name);
context(const char *target_file_name);
~context();
int launch_keops(int tagHostDevice, int dimY, int nx, int ny,
                 int device_id, int tagI, int tagZero, int use_half,
                 int tag1D2D, int dimred,
                 int cuda_block_size, int use_chunk_mode,
                 int *indsi, int *indsj, int *indsp,
                 int dimout,
                 int *dimsx, int *dimsy, int *dimsp,
                 const std::vector<int*>& ranges_v,
                 int *shapeout, void *out_void, int nargs, 
                 const std::vector<void*>& arg_v,
                 const std::vector<int*>& argshape_v
                 );
};