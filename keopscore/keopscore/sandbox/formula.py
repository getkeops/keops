# testing some formulas with keopscore

from keopscore.formulas import *

import keopscore

keopscore.auto_factorize = True

print("********************************")
print("test 1")

f = Grad_WithSavedForward(
    Sum_Reduction(Sum((Var(0, 1, 0) - Var(1, 1, 1))), 1),
    Var(0, 1, 0),
    Var(2, 1, 1),
    Var(3, 1, 1),
)
print(f)

print("********************************")
print("test 2")

x = Var(0, 3, 0, "x")
y = Var(1, 3, 1, "y")
K = Exp(-SqDist(x, y))
Klap = Laplacian(K, y)

print(Klap)

print("********************************")
print("test 3")

x = Var(0, 4, 0, "x")
y = Var(1, 2, 1, "y")
K = Exp(SqNorm2(MatVecMult(x, y)))
Klap = Laplacian(K, y)

print(Klap)
