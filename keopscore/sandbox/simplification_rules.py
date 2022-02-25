# Testing simplification rules of formulas

from keopscore.formulas import *

x = Var(0, 3, 0)
y = Var(1, 3, 1)

f = 3 * (x - y) ** 2 - 2 * (x - y) ** 2 * 2

print()
print("f =", f)
print()
