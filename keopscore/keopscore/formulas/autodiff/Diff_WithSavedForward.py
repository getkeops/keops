# same as Diff with additional saved forward variable. This is only used for taking diff of reductions operations.


def Diff_WithSavedForward(red_formula, v, u, f0):
    return red_formula.Diff(v, u, f0)
