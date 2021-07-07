# N.B. used internally for passing parameters to TensorDot,
# for compatibility with previous C++ KeOps code

import numpy as np

def Ind(*x):
    return np.array(tuple(x)).flatten()