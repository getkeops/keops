# Testing simplification rules of formulas

from keopscore.formulas import *

D = 3
x = Var(0, D, 0, "x")
y = Var(1, D, 1, "y")

f = 3 * (x - y) ** 2 - 2 * (x - y) ** 2 * 2

print("testing simple simplification")
print("f =", f)
print()

print("*******************")

Dv = 3
b = Var(2, Dv, 1, "b")
f = SqDist(x, y) * b

print("testing Grad and Diff")
print("f =", f)
print()

e = Var(3, Dv, 0, "e")
dfT = Grad(f, x, e)
print("dfT=", dfT)

u = Var(4, D, 0, "u")
df = Diff(f, x, u)
print("df=", df)

print("<dfT,u>", dfT | u)
print("<df,e>", df | e)

print("<df,e>-<dfT,u>", (df | e) - (dfT | u))

print("*******************")

divf = TraceOperator(dfT, e)
print("divf=", divf)

divf = TraceOperator(df, u)
print("divf=", divf)

divf = Divergence(f, x)
print("divf=", divf)
