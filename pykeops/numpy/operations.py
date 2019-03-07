from pykeops.common.operations import softmax as softmax_common

def softmax(formula,formula_weights,variables,dtype='float64'):
    return softmax_common(formula,formula_weights,variables,'numpy',dtype)
