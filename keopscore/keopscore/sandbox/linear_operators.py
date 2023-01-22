from keopscore.formulas import *

D = 3
x = Var(0, D, 0, "x")
y = Var(1, D, 1, "y")
# f = Exp(-SqDist(x,y))
f = Exp(-Sum((x - y) ** 2) / 2)

print("f=", f)

u = Var(2, 1, 0, "u")
dfT = Grad(f, x, 1)

print("dfT=", dfT)

w = Var(3, D, 0, "w")
h = Grad(dfT, x, w)

print("h=", h)

lap = TraceOperator(h, w)

print("lap=", lap)

lap2 = Laplacian(f, x)

print("lap2=", lap2)

lap3 = AutoFactorize(lap2)

print("lap3=", lap3)
