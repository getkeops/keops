###########################################################
#              Compilation options
###########################################################

default_cuda_type = 'float'
dll_prefix = "lib"
dll_ext = ".so"

############################################################
#              Search for GPU
############################################################

# is there a working GPU around ?
import GPUtil
try:
    gpu_available = len(GPUtil.getGPUs()) > 0
except:
    gpu_available = False

# is torch installed ?
try:
    import torch
    torch_found = True
    gpu_available = torch.cuda.is_available() # if torch is found, we use it to detect the gpu

except:
    torch_found = False

############################################################
#     define backend
############################################################
import re

class pykeops_backend():
    """
    This class is  used to centralized the options used in PyKeops.
    """

    dev_list = ['CPU','GPU']
    grid_list = ['1D','2D']
    memtype_list = ['host','device']
    backend_list = ['CPU','GPU_1D_device','GPU_1D_host','GPU_2D_device','GPU_2D_host']

    possible_options_list = ["auto", "CPU", "GPU", "GPU_1D", 'GPU_1D_device','GPU_1D_host', "GPU_2D",'GPU_2D_device','GPU_2D_host']

    def define_backend(self,backend,result,variables):
        """
         Try to make a good guess for the backend...  available methods are: (host means Cpu, device means Gpu)
           CPU : computations performed with the host from host arrays
           GPU_1D_device : computations performed on the device from device arrays, using the 1D scheme
           GPU_2D_device : computations performed on the device from device arrays, using the 2D scheme
           GPU_1D_host : computations performed on the device from host arrays, using the 1D scheme
           GPU_2D_host : computations performed on the device from host data, using the 2D scheme

        """

        # check that the option is valid
        if (backend not in self.possible_options_list):
            raise ValueError('Invalid backend specified. Should be one of ', self.possible_options_list)

        # if the option is known: exit.
        if (backend in self.backend_list):
            return backend

        # else  try to infer the missing values
        if (backend == 'auto'):
            dev_type = self._find_dev()
            grid_type = self._find_grid()
            mem_type = self._find_mem(result,variables)
            return dev_type+'_'+grid_type+'_'+mem_type if dev_type == 'GPU' else dev_type

        splitted_backend = re.split('_',backend)

        if len(splitted_backend) == 1: # GPU_1D or GPU_2D
            grid_type = self._find_grid()
            mem_type = self._find_mem(result,variables)
            return backend+'_'+grid_type+'_'+mem_type

        elif len(splitted_backend) == 2: # CPU or GPU
            mem_type = self._find_mem(result,variables)
            return backend+'_'+mem_type
        

    def _find_dev(self):
            from pykeops import gpu_available
            return self.dev_list[1] if gpu_available else self.dev_list[0]


    def _find_mem(self,result,variables):
        # Infer if we're working with numpy arrays or torch tensors from result's type :
        if hasattr(result, "ctypes"):  # Assume we're working with numpy arrays
            from pykeops.numpy.utils import is_on_device
        elif hasattr(result, "data_ptr"):  # Assume we're working with torch tensors
            from pykeops.torch.utils import is_on_device
        else:
            raise TypeError("result should either be a numpy array or a torch tensor.")

        # first determine where is located the data ; all arrays should be on the host or all on the device  
        VarsAreOnGpu = tuple(map(is_on_device,(result,)+tuple(variables))) 
        if all(VarsAreOnGpu):
            MemType = self.memtype_list[1]
        elif not any(VarsAreOnGpu):
            MemType = self.memtype_list[0]
        else:
            raise ValueError("At least two input variables have different memory locations (Cpu/Gpu).")

        return MemType

        
    def _find_grid(self):
        return  self.grid_list[0]

def get_backend(backend,result,variables):
    """
    entry point to get the correct backend
    """
    res = pykeops_backend()
    return res.define_backend(backend,result,variables)

