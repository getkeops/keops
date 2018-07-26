from pykeops import default_cuda_type
from pykeops.common.keops_io import load_keops
from pykeops.common.get_options import get_tag_backend
from pykeops.common.utils import axis2cat


class generic_sum:
    def __init__(self, formula, aliases, axis=0, backend = "auto", cuda_type=default_cuda_type):
        self.formula = formula
        self.aliases = aliases
        self.axis = axis
        self.backend = backend
        self.cuda_type = cuda_type

    def __call__(self, *args) :
        return genred(self.formula, self.aliases, *args, axis=self.axis, backend=self.backend, cuda_type=self.cuda_type)


class generic_logsumexp:
    def __init__(self, formula, aliases, axis=0, backend = "auto", cuda_type=default_cuda_type):
        self.formula ="LogSumExp(" + formula + ")"
        self.aliases = aliases
        self.axis = axis
        self.backend = backend
        self.cuda_type = cuda_type

    def __call__(self, *args):
        return genred(self.formula, self.aliases, *args, axis=self.axis, backend=self.backend, cuda_type=self.cuda_type)


def genred(formula, aliases, *args, axis = 0, backend="auto", cuda_type=default_cuda_type):
    myconv = load_keops(formula, aliases, cuda_type, 'numpy')

    # Get tags
    tagIJ =  axis2cat(axis)  # tagIJ=0 means sum over j, tagIJ=1 means sum over j
    tagCpuGpu, tag1D2D, _ = get_tag_backend(backend, args)

    # Perform computation using KeOps
    result = myconv.genred_numpy(tagIJ, tagCpuGpu, tag1D2D, 0, *args)  # the extra zeros is mandatory but has no effect

    # import numpy as np
    # args2 = np.copy(args)
    # args2[2] = np.ascontiguousarray(np.copy(args[2].T)).T
    # print(args[2].flags.c_contiguous)
    # print(args2[2].flags.c_contiguous)
    # result2 = myconv.genred_numpy(tagIJ, tagCpuGpu, tag1D2D, 0, args2[0], args2[1], args2[2], args2[3] )  # the extra zeros is mandatory but has no effect

    return result