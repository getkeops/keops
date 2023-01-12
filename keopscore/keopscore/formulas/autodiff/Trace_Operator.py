from keopscore.formulas import Var
from keopscore.utils.code_gen_utils import GetInds
from keopscore.utils.misc_utils import KeOps_Error
from keopscore.formulas.maths.Minus import Minus_Impl
from keopscore.formulas.maths.Add import Add_Impl
from keopscore.formulas.maths.Subtract import Subtract_Impl
from keopscore.formulas.maths.Mult import Mult_Impl
from keopscore.formulas.maths.Divide import Divide_Impl
from keopscore.formulas.maths.Exp import Exp
from keopscore.formulas.variables.IntCst import IntCst_Impl
from keopscore.formulas.variables.Var import Var
from keopscore.formulas.variables.Zero import Zero
from keopscore.formulas.maths.Scalprod import Scalprod_Impl

# /////////////////////////////////////////////////////////////
# ///      TRACE OPERATOR  : Trace_Operator< F, V >       ////
# /////////////////////////////////////////////////////////////


def Trace_Operator(formula, v, w=1):
    
    if isinstance(formula,Minus_Impl):
        fa, = formula.children
        return -Trace_Operator(fa,v,w)
    elif isinstance(formula,Subtract_Impl):
        fa, fb = formula.children
        return Trace_Operator(fa,v,w)-Trace_Operator(fb,v,w)
    elif isinstance(formula,Add_Impl):
        fa, fb = formula.children
        return Trace_Operator(fa,v,w)+Trace_Operator(fb,v,w)
    elif isinstance(formula,Mult_Impl):
        fa, fb = formula.children
        if fa.dim==1:
            xxx
        return Trace_Operator(fa,v,fb*w) + Trace_Operator(fb,v,fa*w)
    elif isinstance(formula,Divide_Impl):
        fa, fb = formula.children
        return Trace_Operator(fa,v,w/fb)
    elif isinstance(formula,Var):
        return IntCst_Impl(v.dim) if v==formula else formula
    elif isinstance(Scalprod_Impl,Var):
        fa, fb = formula.children
        return Trace_Operator(fa,v,w)|Trace_Operator(fb,v,w)
    else:
        print(type(formula))
        raise ValueError("not implemented")
