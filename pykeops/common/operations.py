

def softmax(formula,formula_weights,variables,backend,dtype='float32'):
    if backend=='numpy':
        from pykeops.numpy import Genred
    elif backend=='torch':
        from pykeops.torch import Genred
    formula2 = 'Concat(IntCst(1),' + formula_weights + ')'
    my_routine = Genred(formula, variables, reduction_op='LogSumExpVect', axis=1, cuda_type=dtype, formula2=formula2)
    def f(*args):
        out = my_routine(*args, backend="auto")
        out = out[:,2:]/out[:,1][:,None]
        return out
    return f
