# same as Grad with additional saved forward variable. This is only used for taking gradients of reductions operations.


def Grad_WithSavedForward(red_formula, v, gradin, f0):
    return red_formula.DiffT(v, gradin, f0)
