from keops.python_engine.formulas.maths.Sum import Sum

##########################
#####    Scalprod     ####
##########################

def Scalprod(arg0, arg1):
    if arg0.dim == 1:
        return arg0 * Sum(arg1)
    elif arg1.dim == 1:
        return Sum(arg0) * arg1
    else:
        return Sum(arg0 * arg1)