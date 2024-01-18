# Testing auto factorize for high order grads

from keopscore.formulas import *

D = 3
x = Var(0, D, 0, "x")
y = Var(1, D, 1, "y")

Dv = 3
b = Var(2, Dv, 1, "b")
f = SqNorm2(Exp(-SqDist(x, y)) * b)

print("f =", f)
print()
g = AutoFactorize(f)

for i in range(1, 7):
    print("grad order", i)
    f = Grad(f, x)
    g = AutoFactorize(Grad(g, x))
    print()
    print(f)
    print()
    print("with auto factorize :")
    print()
    print(AutoFactorize(f))
    print()
    print("with auto factorize iterated:")
    print()
    print(g)
    print()


f = SqNorm2(Exp(-SqDist(x, y)) * b)

print("f =", f)
print()
g = AutoFactorize(f)

for i in range(1, 9):
    print("grad order", i)
    g = AutoFactorize(Grad(g, x))
    print()
    print("with auto factorize iterated:")
    print()
    print(g)
    print()
