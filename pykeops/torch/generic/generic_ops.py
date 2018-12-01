from pykeops.common.utils import cat2axis
from pykeops.common.parse_type import get_type
from pykeops.torch.generic.generic_red import Genred


def generic_sum(formula, output, *aliases, **kwargs) :
    _,cat,_,_ = get_type(output)
    axis = cat2axis(cat)
    return Genred(formula, aliases, reduction_op='Sum', axis=axis, **kwargs)

def generic_logsumexp(formula, output, *aliases, **kwargs) :
    _,cat,_,_ = get_type(output)
    axis = cat2axis(cat)
    return Genred(formula, aliases, reduction_op='LogSumExp', axis=axis,  **kwargs)

def generic_argkmin(formula, output, *aliases, **kwargs) :
    _,cat,_,_ = get_type(output)
    axis = cat2axis(cat)
    return Genred(formula, aliases, reduction_op='ArgKMin', axis=axis,  **kwargs)


def generic_argmin(formula, output, *aliases, **kwargs) :
    _,cat,_,_ = get_type(output)
    axis = cat2axis(cat)
    return Genred(formula, aliases, reduction_op='ArgMin', axis=axis,  **kwargs)
