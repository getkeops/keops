from keopscore.formulas import *

x = Var(0, 3, 0)
y = Var(1, 3, 1)
e = Exp(-SqDist(x, y))

K1 = e * (e - 1)
K2 = AutoFactorize(K1)

print(K1)
print(K2)


print("****")

GK1 = Laplacian(K1, x)
GK2 = Laplacian(K2, x)

print(GK1)
print(GK2)

print("****")

GK12 = AutoFactorize(GK1)
GK22 = AutoFactorize(GK2)

print(GK12)
print(GK22)
