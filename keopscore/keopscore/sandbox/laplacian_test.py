from keopscore.formulas import *

D = 3
x = Var(0, D, 0, "x")
y = Var(1, D, 1, "y")
# f = Exp(-SqDist(x,y))
f = Exp(-Sum((x - y) ** 2) / 2)

print("\nf=", f)

dfT = Grad(f, x, 1)

print("\ndfT=", dfT)

h = GradMatrix(dfT, x)

print("\nh=", h)

h_ = AutoFactorize(h)

print("\nh_=", h_)

print("****")

g = Concat(x + y, x - y)
print(g)

print("****")

g = Elem(x + y, 0)
print(g)
