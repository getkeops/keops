
import numpy as np

from pykeops.numpy import Genred

from pykeops import default_cuda_type
from pykeops.common.utils import axis2cat, cat2axis
from pykeops.common.parse_type import get_type, get_sizes, complete_aliases
from pykeops.common.get_options import get_tag_backend
from pykeops.common.keops_io import load_keops

include_dirs = torch.utils.cpp_extension.include_paths()[0]

from pykeops.tutorials.interpolation.common.linsolve import ConjugateGradientSolver

backend = np
copy = np.copy
tile = np.tile
solve = np.linalg.solve
norm = np.linalg.norm
Genred = GenredNumpy
rand = lambda m, n : np.random.rand(m,n,dtype=dtype)
randn = lambda m, n : np.random.randn(m,n,dtype=dtype)
zeros = lambda shape : np.zeros(shape,dtype=dtype)
eye = lambda n : np.eye(n,dtype=dtype)
array = lambda x : np.array(x,dtype=dtype)
arraysum = np.sum
transpose = lambda x : x.T
numpy = lambda x : x
        

class InvKernelOp:
    
    def __init__(self, formula, aliases, varinvalias, reduction_op='Sum', axis=0, cuda_type=default_cuda_type, opt_arg=None):
        if opt_arg:
            self.formula = reduction_op + 'Reduction(' + formula + ',' + str(opt_arg) + ',' + str(axis2cat(axis)) + ')'
        else:
            self.formula = reduction_op + 'Reduction(' + formula + ',' + str(axis2cat(axis)) + ')'
        self.aliases = complete_aliases(formula, aliases)
        self.varinvalias = varinvalias
        self.cuda_type = cuda_type
        self.myconv = load_keops(self.formula,  self.aliases,  self.cuda_type, 'numpy')

        tmp = aliases.copy()
        for (i,s) in enumerate(tmp):
            tmp[i] = s[:s.find("=")].strip()
        self.varinvpos = tmp.index(varinvalias)

    def __call__(self, *args, backend='auto', device_id=-1):
        # Get tags
        tagCpuGpu, tag1D2D, _ = get_tag_backend(backend, args)
        nx, ny = get_sizes(self.aliases, *args)
        dtype = args[0].dtype.name
        varinv = args[varinvpos]      
        def linop(var):
            newargs = args[:varinvpos] + (var,) + args[varinvpos+1:]
            return myconv.genred_numpy(nx, ny, tagCPUGPU, tag1D2D, 0, device_id, *newargs)
        return ConjugateGradientSolver(linop,varinv,eps=1e-16)
     
     
     



