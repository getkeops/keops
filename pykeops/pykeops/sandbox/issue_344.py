import numpy as np
from pykeops.numpy import Genred

formula = "Sum(V0-V1)"
variables = ["V0=Vi(3)", "V1=Vj(3)", "V2=Vj(3)"]

# Compile corresponding operator
op = Genred(formula, variables, reduction_op="LogSumExpWeight", axis=0, formula2="V2")

# data
x = np.random.randn(4, 3)
y = np.random.randn(5, 3)
w = np.ones([5, 3])

# Perform reduction
res = op(x, y, w, backend="CPU")
