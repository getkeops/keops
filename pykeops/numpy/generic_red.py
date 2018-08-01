from pykeops import default_cuda_type
from pykeops.common.keops_io import load_keops
from pykeops.common.get_options import get_tag_backend
from pykeops.common.utils import axis2cat


class generic_red:
    def __init__(self, formula, aliases, reduction_op="Sum", axis=1, backend = "auto", cuda_type=default_cuda_type):
        self.formula = reduction_op + "Reduction(" + formula + "," + str(axis2cat(axis)) + ")"
        self.aliases = aliases
        self.backend = backend
        self.cuda_type = cuda_type

    def __call__(self, *args) :
        return genred(self.formula, self.aliases, *args, backend=self.backend, cuda_type=self.cuda_type)


def genred(formula, aliases, *args, backend="auto", cuda_type=default_cuda_type):
    
    myconv = load_keops(formula, aliases, cuda_type, 'numpy')
    
    # Get tags
    tagCpuGpu, tag1D2D, _ = get_tag_backend(backend, args)

    # Perform computation using KeOps
    result = myconv.genred_numpy(tagCpuGpu, tag1D2D, 0, *args)  # the extra zeros is mandatory but has no effect

    # import numpy as np
    # args2 = np.copy(args)
    # args2[2] = np.ascontiguousarray(np.copy(args[2].T)).T
    # print(args[2].flags.c_contiguous)
    # print(args2[2].flags.c_contiguous)
    # result2 = myconv.genred_numpy(tagIJ, tagCpuGpu, tag1D2D, 0, args2[0], args2[1], args2[2], args2[3] )  # the extra zeros is mandatory but has no effect

    return result