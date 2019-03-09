import numpy as np

from pykeops.common.utils import axis2cat, cat2axis
from pykeops.numpy import default_cuda_type
from pykeops.common.parse_type import get_type, get_sizes, complete_aliases
from pykeops.common.get_options import get_tag_backend
from pykeops.common.keops_io import load_keops

from pykeops.common.operations import Genred_common
def Genred(formula, aliases, reduction_op='Sum', axis=0, cuda_type=default_cuda_type, opt_arg=None, formula2=None):
    return Genred_common('numpy', formula, aliases, reduction_op, axis, cuda_type, opt_arg, formula2)


from pykeops.common.operations import ConjugateGradientSolver        
class InvKernelOp:
    
    def __init__(self, formula, aliases, varinvalias, lmbda=0, axis=0, dtype=default_cuda_type, opt_arg=None):
        reduction_op='Sum'
        if opt_arg:
            self.formula = reduction_op + 'Reduction(' + formula + ',' + str(opt_arg) + ',' + str(axis2cat(axis)) + ')'
        else:
            self.formula = reduction_op + 'Reduction(' + formula + ',' + str(axis2cat(axis)) + ')'
        self.aliases = complete_aliases(formula, aliases)
        self.varinvalias = varinvalias
        self.dtype = dtype
        self.myconv = load_keops(self.formula,  self.aliases,  self.dtype, 'numpy')
        self.lmbda = lmbda
        tmp = aliases.copy()
        for (i,s) in enumerate(tmp):
            tmp[i] = s[:s.find("=")].strip()
        self.varinvpos = tmp.index(varinvalias)

    def __call__(self, *args, backend='auto', device_id=-1, eps=1e-6):
        # Get tags
        tagCpuGpu, tag1D2D, _ = get_tag_backend(backend, args)
        nx, ny = get_sizes(self.aliases, *args)
        varinv = args[self.varinvpos]      
        def linop(var):
            newargs = args[:self.varinvpos] + (var,) + args[self.varinvpos+1:]
            res = self.myconv.genred_numpy(nx, ny, tagCpuGpu, tag1D2D, 0, device_id, *newargs)
            if self.lmbda:
                res += self.lmbda*var
            return res
        return ConjugateGradientSolver('numpy',linop,varinv,eps=eps)
     
     
     



