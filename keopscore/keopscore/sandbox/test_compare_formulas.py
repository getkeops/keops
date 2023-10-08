import torch
from keopscore.utils.TestFormula import TestFormula
from keopscore.formulas import *

randseed = torch.randint(1000, (1,)).item()

x = Var(0, 3, 0)
y = Var(1, 3, 1)
b = Var(2, 1, 1)

f = Exp(-Sum((x - y) ** 2)) * b

dx1 = Var(3, x.dim, 0)
dx2 = Var(4, x.dim, 0)

# hessian as diff of diff
df = Diff(f, x, dx1)
formula1 = Diff(df, x, dx2)

# hessian as diff of grad
gf = Grad(f, x, IntCst(1))
formula2 = dx2 | Diff(gf, x, dx1)

res1 = TestFormula(formula1, randseed=randseed)
res2 = TestFormula(formula2, randseed=randseed)
print("relative error:", (torch.norm(res1 - res2) / torch.norm(res1)).item())

# hessian as grad of diff
formula3 = dx2 | Grad(df, x, IntCst(1))
res3 = TestFormula(formula2, randseed=randseed)
print("relative error:", (torch.norm(res1 - res3) / torch.norm(res1)).item())
