# Testing auto factorize

from keopscore.formulas import *

D = 3
x = Var(0, D, 0, "x")
y = Var(1, D, 1, "y")

Dv = 3
b = Var(2, Dv, 1, "b")
f = SqNorm2(Exp(-SqDist(x, y)) * b) + b

g = AutoFactorize(f)

print("f =", f)
print("g =", g)
print()
