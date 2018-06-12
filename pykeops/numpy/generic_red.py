import numpy as np

from pykeops.common.utils import parse_types, create_name
from pykeops.common.generic_reduction import genred

class generic_sum_np :
    def __init__(self, formula, *types) :
        self.formula = formula
        self.aliases, self.sum_index = parse_types( types )

    def __call__(self, *args, backend = "auto") :
        return genred(self.formula, self.aliases, *args, sum_index = self.sum_index,backend = backend)


class generic_logsumexp_np :
    def __init__(self, formula, *types) :
        self.formula ="LogSumExp(" + formula + ")"
        self.aliases, self.sum_index = parse_types( types )
        
    def __call__(self, *args, backend = "auto") :
        return genred(self.formula, self.aliases, *args, sum_index = self.sum_index,backend = backend)


