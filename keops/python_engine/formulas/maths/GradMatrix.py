from keops.python_engine.formulas.maths.ElemT import ElemT
from keops.python_engine.formulas.variables.IntCst import IntCst


# //////////////////////////////////////////////////////////////////////////////////////////////
# ////      Standard basis of R^DIM : < (1,0,0,...) , (0,1,0,...) , ... , (0,...,0,1) >     ////
# //////////////////////////////////////////////////////////////////////////////////////////////

def StandardBasis_Impl(dim):
    return tuple(ElemT(IntCst(1), dim, i) for i in range(dim))


# /////////////////////////////////////////////////////////////////////////
# ////      Matrix of gradient operator (=transpose of jacobian)       ////
# /////////////////////////////////////////////////////////////////////////

"""
def GradMatrix_Impl(f, v):
    # TODO : finish this
    f.Vars(cat=3)
    
     {
  using IndsTempVars = GetInds<typename F::template VARS<3>>;
  using GRADIN = Var<1 + IndsTempVars::MAX, F::DIM, 3>;
  using packGrads = IterReplace <Grad<F, V, GRADIN>, GRADIN, StandardBasis<F::DIM>>;
  using type = IterBinaryOp<Concat, packGrads>;
};


#define GradMatrix(f,g) KeopsNS<GradMatrix<decltype(InvKeopsNS(f)),decltype(InvKeopsNS(g))>>()

}
"""