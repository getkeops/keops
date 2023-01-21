# Testing simplification rules of formulas

from keopscore.formulas import *

D = 3
x = Var("x", D, 0)
y = Var("y", D, 1)

f = 3 * (x - y) ** 2 - 2 * (x - y) ** 2 * 2

print("testing simple simplification")
print("f =", f)
print()

Dv = 2
b = Var("b", Dv, 1)
f = SqDist(x,y)*b

print("testing Grad and Diff")
print("f =", f)
print()

e = Var("e", Dv, 0)
dfT = Grad(f,x,e)
print("dfT=", dfT)

u = Var("u", D, 0)
df = Diff(f,x,u)
print("df=", df)

print("<dfT,u>", dfT|u)
print("<df,e>", df|e)

print("<df,e>-<dfT,u>", (df|e)-(dfT|u))


