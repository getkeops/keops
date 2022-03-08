from keopscore.formulas import *

f = Grad_WithSavedForward(
    Sum_Reduction(Sum((Var(0, 1, 0) - Var(1, 1, 1))), 1),
    Var(0, 1, 0),
    Var(2, 1, 1),
    Var(3, 1, 1),
)
print(f)
