from keopscore.formulas import *

D = 3
x = Var(0, D, 0)
y = Var(1, D, 1)

Dv = 3
b = Var(2, Dv, 1)
f = b**2 * SqDist(x, y) * Exp(-SqDist(x, y)) * (x - y) * b**2

print()
print("f=", f)

g = AutoFactorize(f)

print()
print("g=", g)
