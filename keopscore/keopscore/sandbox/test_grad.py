# Testing grad engine

from keopscore.formulas import *
from keopscore.utils.TestFormula import TestFormula
D, Dv = 3, 2
x = Var(0, D, 0, "x")
y = Var(1, D, 1, "y")
b = Var(2, Dv, 1, "b")

f = Exp(-SqDist(x, y)) * b
print("f =", f)

u = Var(3, Dv, 0, "u")

g = Concat(Grad(f,x,u),Grad(f,b,u))
print("g=",g)

TestFormula(g, randseed=1)

g = AutoFactorize(g)
print("g=",g)

TestFormula(g, randseed=1)

g = Grad(f,[x,b],u)
print("g=",g)

TestFormula(g, randseed=1)

g = AutoFactorize(g)
print("g=",g)
TestFormula(g, randseed=1)