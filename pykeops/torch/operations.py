from pykeops.common.operations import softmax as softmax_common

def softmax(formula,formula_weights,variables,dtype='float32'):
    return softmax_common(formula,formula_weights,variables,'torch',dtype)
