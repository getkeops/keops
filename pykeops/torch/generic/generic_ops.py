from pykeops.common.utils import cat2axis
from pykeops.common.parse_type import get_type

from pykeops.torch.generic.generic_red import Genred

def generic_sum(formula, *aliases, **kwargs) :
    _,cat,_,_ = get_type(aliases[0])
    axis = cat2axis(cat)
    return Genred(formula, aliases[1:], reduction_op='Sum', axis=axis, **kwargs)

def generic_logsumexp(formula, *aliases, **kwargs) :
    _,cat,_,_ = get_type(aliases[0])
    axis = cat2axis(cat)
    routine = Genred(formula, aliases[1:], reduction_op='LogSumExp', axis=axis,  **kwargs)
    def red_routine(*args, **kwargs) :
        tmp = routine(*args, **kwargs)
        return tmp[:,0] + (tmp[:,1]).log()
    return red_routine

def generic_argkmin(formula, *aliases, **kwargs) :
    _,cat,_,_ = get_type(aliases[0])
    axis = cat2axis(cat)
    return Genred(formula, aliases[1:], reduction_op='ArgKMin', axis=axis,  **kwargs)

def generic_argmin(formula, *aliases, **kwargs) :
    _,cat,_,_ = get_type(aliases[0])
    axis = cat2axis(cat)
    return Genred(formula, aliases[1:], reduction_op='ArgMin', axis=axis,  **kwargs)
