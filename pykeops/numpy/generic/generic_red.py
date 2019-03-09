from pykeops.common.keops_io import load_keops
from pykeops.common.get_options import get_tag_backend
from pykeops.common.parse_type import get_sizes, complete_aliases
from pykeops.common.utils import axis2cat


class Genred_lowlevel:
    def __init__(self, formula, aliases, reduction_op, axis, cuda_type, opt_arg, formula2):
        str_opt_arg = ',' + str(opt_arg) if opt_arg else ''
        str_formula2 = ',' + formula2 if formula2 else ''     
        self.formula = reduction_op + 'Reduction(' + formula + str_opt_arg + ',' + str(axis2cat(axis)) + str_formula2 + ')'
        self.aliases = complete_aliases(self.formula, aliases)
        self.cuda_type = cuda_type
        self.myconv = load_keops(self.formula,  self.aliases,  self.cuda_type, 'numpy')

    def __call__(self, *args, backend, device_id):
        # Get tags
        tagCpuGpu, tag1D2D, _ = get_tag_backend(backend, args)
        nx, ny = get_sizes(self.aliases, *args)
        return self.myconv.genred_numpy(nx, ny, tagCpuGpu, tag1D2D, 0, device_id, *args) 
