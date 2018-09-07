from pykeops import default_cuda_type
from pykeops.common.keops_io import load_keops
from pykeops.common.get_options import get_tag_backend
from pykeops.common.utils import axis2cat


class Genred:
    def __init__(self, formula, aliases, reduction_op='Sum', axis=1, cuda_type=default_cuda_type, opt_arg=None):
        if opt_arg:
            self.formula = reduction_op + 'Reduction(' + formula + ',' + str(opt_arg) + ',' + str(axis2cat(axis)) + ')'
        else:
    	    self.formula = reduction_op + 'Reduction(' + formula + ',' + str(axis2cat(axis)) + ')'
        self.aliases = aliases
        self.cuda_type = cuda_type
        self.myconv = load_keops(self.formula,  self.aliases,  self.cuda_type, 'numpy')

    def __call__(self, *args, backend='auto'):
        # Get tags
        tagCpuGpu, tag1D2D, _ = get_tag_backend(backend, args)

        return self.myconv.genred_numpy(tagCpuGpu, tag1D2D, 0, *args) # the extra zeros is mandatory but has no effect