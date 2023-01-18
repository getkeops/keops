from keopscore.formulas import *

D = 3
x = Var(0,D,0)
y = Var(1,D,1)
#f = Exp(-SqDist(x,y))
f = Exp(-Sum((x-y)**2)/2)

print("f=",f)

u = Var(2,1,0)
dfT = Grad(f,x,1)

print("dfT=",dfT)

#v = Var(3,D,0)
#df = AdjointOperator(dfT,u,v)
#print("df=",df)

w = Var(4,D,0)
h = Grad(dfT,x,w)

print("h=",h)

lap = TraceOperator(h,w)

print("lap=",lap)

lap2 = Laplacian(f,x)

print("lap2=",lap2)