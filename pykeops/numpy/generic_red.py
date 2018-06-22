from pykeops.common.generic_reduction import genred
from pykeops import default_cuda_type

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


