# Testing simplification rules of formulas

from keopscore.formulas import *
import timeit

def g():
    x = Var(0, 3, 0)
    y = Var(1, 3, 1)
    return 3 * (x - y) ** 2 - 2 * (x - y) ** 2 * 2

print("elapsed = ", timeit.timeit(g, number=100))

