import numpy as np
from pykeops.numpy import Vi, Vj, Pm

Cplx = 2 + 3j
v = [2 + 3j, 7 + 3j, 4 + 1j]
x = np.random.rand(10, 2)

Pm_Scal = Pm(3.14)
Pm_Cplx = Pm(Cplx)
Pm_v = Pm(v)
x_i = Vi(x)

print(Pm_v + Pm_Scal)  # ValueError: incompatible shapes for addition.
print(Pm_v + Pm_Cplx)  # ValueError: incompatible shapes for addition.
print(x_i + Pm_Cplx)  # ValueError: incompatible shapes for addition.
