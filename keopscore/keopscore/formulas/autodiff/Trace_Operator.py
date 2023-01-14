from keopscore.formulas import Var
from keopscore.utils.code_gen_utils import GetInds
from keopscore.utils.misc_utils import KeOps_Error
from keopscore.formulas.maths.Minus import Minus_Impl
from keopscore.formulas.maths.Add import Add_Impl
from keopscore.formulas.maths.Subtract import Subtract_Impl
from keopscore.formulas.maths.Mult import Mult_Impl
from keopscore.formulas.maths.Sum import Sum_Impl, Sum
from keopscore.formulas.maths.SumT import SumT_Impl
from keopscore.formulas.maths.Divide import Divide_Impl
from keopscore.formulas.maths.Exp import Exp
from keopscore.formulas.variables.IntCst import IntCst_Impl
from keopscore.formulas.variables.Var import Var
from keopscore.formulas.variables.Zero import Zero
from keopscore.formulas.maths.Scalprod import Scalprod_Impl
from keopscore.formulas.Operation import Broadcast

# /////////////////////////////////////////////////////////////
# ///      TRACE OPERATOR  : Trace_Operator< F, V >       ////
# /////////////////////////////////////////////////////////////

def Sum_Lin_Operator(formula, v, w=1):
    if not formula.is_linear(v):
        raise ValueError("Formula is not linear with respect to variable.")
    if type(formula) in [Add_Impl, Minus_Impl]:
        newargs = (Sum_Lin_Operator(Broadcast(f,v.dim),v,w) for f in formula.children)
        return type(formula)(*newargs, *formula.params)
    elif isinstance(formula,Mult_Impl):
        fa, fb = formula.children
        if fa.is_linear(v):
            fa = Broadcast(fa,v.dim)
            return Sum_Lin_Operator(fa,v,fb*w)
        elif fb.is_linear(v):
            fb = Broadcast(fb,v.dim)
            return Sum_Lin_Operator(fb,v,fa*w)
    elif isinstance(formula,Divide_Impl):
        fa, fb = formula.children
        fa = Broadcast(fa,v.dim)
        return Sum_Lin_Operator(fa,v,w/fb)
    elif isinstance(formula,Var):
        return IntCst_Impl(v.dim)*w if v==formula else formula
    elif isinstance(formula,Scalprod_Impl):
        fa, fb = formula.children
        return Sum_Lin_Operator(Sum(fa*fb),v,w)
    elif isinstance(formula,SumT_Impl):
        f, = formula.children
        return formula.dim*Sum_Lin_Operator(f,v,w)
    elif isinstance(formula,Sum_Impl):
        f, = formula.children
        return Sum_Lin_Operator(f,v,w)
    else:
        raise ValueError(f"Sum_Lin_Operator not implemented for {formula.string_id}")
    

def Trace_Operator(formula, v, w=1):
    
    if not formula.is_linear(v):
        raise ValueError("Formula is not linear with respect to variable.")
    
    if formula.dim != v.dim:
        raise ValueError("Input and Output dimensions should match.")
        
    if type(formula) in [Add_Impl, Minus_Impl]:
        newargs = (Trace_Operator(Broadcast(f,v.dim),v,w) for f in formula.children)
        return type(formula)(*newargs, *formula.params)
    elif isinstance(formula,Mult_Impl):
        fa, fb = formula.children
        if fa.is_linear(v):
            fa = Broadcast(fa,v.dim)
            return Trace_Operator(fa,v,fb*w)
        elif fb.is_linear(v):
            fb = Broadcast(fb,v.dim)
            return Trace_Operator(fb,v,fa*w)
    elif isinstance(formula,Divide_Impl):
        fa, fb = formula.children
        fa = Broadcast(fa,v.dim)
        return Trace_Operator(fa,v,w/fb)
    elif isinstance(formula,Var):
        return IntCst_Impl(v.dim)*w if v==formula else formula
    elif isinstance(formula,Scalprod_Impl):
        fa, fb = formula.children
        return Trace_Operator(Sum(fa*fb),v,w)
    elif isinstance(formula,SumT_Impl):
        f, = formula.children
        return Sum_Lin_Operator(f,v,w)
    else:
        raise ValueError(f"Trace_Operator not implemented for {formula.string_id}")
