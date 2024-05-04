# Testing auto factorize

from keopscore.formulas import *
from keopscore.formulas.factorization.Factorize import AutoFactorize, UnFactorize
from time import time

D = 3
Dv = 3
x = Var(0, D, 0, "x")
y = Var(1, D, 1, "y")
b = Var(2, Dv, 1, "b")
u = Var(3, D, 1, "u")

# f = SqNorm2(Exp(-SqDist(x, 2*y)) * b) + b + 2*y + SqDist(x, 2*y)
f = Exp(-SqDist(x, y)) * b

start = time()
g = f
for k in range(4):
    g = AutoFactorize(g)
    g = g.DiffT(x, u)
h = AutoFactorize(g)
end = time()
print("elapsed:", end - start)
print("h =", h)
h = UnFactorize(h)
assert h == g, "not good..."
print("h =", h)
