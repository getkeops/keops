import keopscore
from keopscore.formulas import *

D = 3
x = Var(0,D,0)
y = Var(1,D,1)
#f = Exp(-SqDist(x,y))
f = Exp(-Sum((x-y)**2))

print(f)

u = Var(3,1,0)
g = Grad(f,x,u)

v = Var(4,D,0)
h = Grad(g,x,v)

print(h)

print(g.is_linear(u))

print(h.is_linear(u))
print(h.is_linear(v))

print(Trace_Operator(h,v))