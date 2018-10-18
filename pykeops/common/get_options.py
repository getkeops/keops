import re
import os.path
import numpy as np
from collections import OrderedDict
from pykeops import __version__, torch_version_required


###########################################################
#             Set build_folder
###########################################################

def set_build_folder():
    """
    This function set a default build folder that contains the python module compiled
    by pykeops.
    """
    bf_source = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '..' + os.path.sep + 'build' + os.path.sep
    bf_home = os.path.expanduser('~')

    if os.path.isdir(bf_source): # assume we are loading from source
        build_folder  = bf_source
    elif os.path.isdir(bf_home): # assume we are ussing wheel and home is accessible
       build_folder = bf_home + os.path.sep + '.cache' + os.path.sep + 'pykeops-' + __version__ + os.path.sep 
    else: 
        import tempfile
        build_folder = tempfile.mkdtemp(prefix='pykeops-' + __version__) + os.path.sep
        
    os.makedirs(build_folder, exist_ok=True)

    return build_folder



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
    from torch.utils import cpp_extension

    if torch.__version__ < torch_version_required:
        raise ImportError('The pytorch version should be ==' + torch_version_required)

    torch_include_path = torch.utils.cpp_extension.include_paths()[0]
    gpu_available = torch.cuda.is_available() # use torch to detect gpu
    torch_found = True
except ImportError as e: # if 
    print('ImportError: pykeops is not compatible with your version of Pytorch.', e)
    torch_found = False
    torch_include_path = '0'
except:
    torch_found = False
    torch_include_path = '0'

############################################################
#     define backend
############################################################

class pykeops_backend():
    """
    This class is  used to centralized the options used in PyKeops.
    """

    dev = OrderedDict([('CPU',0),('GPU',1)])
    grid = OrderedDict([('1D',0),('2D',1)])
    memtype = OrderedDict([('host',0), ('device',1)])

    possible_options_list = ['auto',
                             'CPU',
                             'GPU',
                             'GPU_1D', 'GPU_1D_device', 'GPU_1D_host',
                             'GPU_2D', 'GPU_2D_device', 'GPU_2D_host'
                             ]

    def define_tag_backend(self, backend, variables):
        """
        Try to make a good guess for the backend...  available methods are: (host means Cpu, device means Gpu)
           CPU : computations performed with the host from host arrays
           GPU_1D_device : computations performed on the device from device arrays, using the 1D scheme
           GPU_2D_device : computations performed on the device from device arrays, using the 2D scheme
           GPU_1D_host : computations performed on the device from host arrays, using the 1D scheme
           GPU_2D_host : computations performed on the device from host data, using the 2D scheme

        :param backend (str), variables (tuple)

        :return (tagCPUGPU, tag1D2D, tagHostDevice)
        """

        # check that the option is valid
        if (backend not in self.possible_options_list):
            raise ValueError('Invalid backend. Should be one of ', self.possible_options_list)

        # auto : infer everything
        if backend == 'auto':
            return int(gpu_available), self._find_grid(), self._find_mem(variables)

        split_backend = re.split('_',backend)
        if len(split_backend) == 1:     # CPU or GPU
            return self.dev[split_backend[0]], self._find_grid(), self._find_mem(variables)
        elif len(split_backend) == 2:   # GPU_1D or GPU_2D
            return self.dev[split_backend[0]], self.grid[split_backend[1]], self._find_mem(variables)
        elif len(split_backend) == 3:   # the option is known
            return self.dev[split_backend[0]], self.grid[split_backend[1]], self.memtype[split_backend[2]]

    def define_backend(self, backend, variables):
        tagCPUGPU, tag1D2D, tagHostDevice  = self.define_tag_backend(backend, variables)
        return self.dev[tagCPUGPU], self.grid[tag1D2D], self.memtype[tagHostDevice]

    @staticmethod
    def _find_dev():
        return int(gpu_available)

    @staticmethod
    def _find_mem(variables):
        if all([type(var) is np.ndarray for var in variables ]): # Infer if we're working with numpy arrays or torch tensors:
            MemType = 0
        elif torch_found and all([type(var) in [torch.Tensor, torch.nn.parameter.Parameter] for var in variables ]):

            from pykeops.torch.utils import is_on_device
            VarsAreOnGpu = tuple(map(is_on_device, tuple(variables)))

            if all(VarsAreOnGpu):
                MemType = 1
            elif not any(VarsAreOnGpu):
                MemType = 0
            else:
                raise ValueError('At least two input variables have different memory locations (Cpu/Gpu).')
        else:
            raise TypeError('All variables should either be numpy arrays or torch tensors.')

        return MemType

    @staticmethod
    def _find_grid():
        return 0


def get_tag_backend(backend, variables, str = False):
    """
    entry point to get the correct backend
    """
    res = pykeops_backend()
    if not str:
        return res.define_tag_backend(backend, variables)
    else:
        return res.define_backend(backend, variables)
