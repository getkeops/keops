from keops.python_engine.formulas.Operation import VectorizedScalarOp, Scalprod, Sum, Square

from code_gen_utils import VectCopy, c_array, c_zero_float
from keops.python_engine.formulas.maths.Abs import keops_abs
from keops.python_engine.formulas.Operation import Operation


##################################################
##########                          ##############
##########      Math operations     ##############
##########                          ##############
##################################################






############################
######    Concat       #####
############################

class Concat(Operation):
    string_id = "Concat"

    def __init__(self, arg0, arg1):
        super().__init__(arg0, arg1)
        self.dim = arg0.dim + arg1.dim

    def Op(self, out, table, arg0, arg1):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        out0, out1 = out.split(arg0.dim, arg1.dim)
        return VectCopy(out0, arg0) + VectCopy(out1, arg1)

    def DiffT(self, v, gradin):
        f = self.children[0]
        g = self.children[1]
        return f.Grad(v, Extract(gradin, 0, f.dim)) + g.Grad(v, Extract(gradin, f.dim, g.dim))


# //////////////////////////////////////////////////////////////
# ////     VECTOR EXTRACTION : Extract<F,START,DIM>         ////
# //////////////////////////////////////////////////////////////

class Extract(Operation):
    string_id = "Extract"

    def __init__(self, arg0, start, dim):
        if arg0.dim < start + dim or start < 0:
            raise ValueError("Index out of bound in Extract")
        super().__init__(arg0)
        self.start = start
        self.dim = dim
        self.params = (start, dim)

    def Op(self, out, table, arg0):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        v = c_array(arg0.dtype, out.dim, f"({arg0.id}+{self.start})")
        return VectCopy(out, v)

    def DiffT(self, v, gradin):
        f = self.children[0]
        return f.Grad(v, ExtractT(gradin, self.start, f.dim))

# //////////////////////////////////////////////////////////////
# ////     VECTOR "INJECTION" : ExtractT<F,START,DIM>       ////
# //////////////////////////////////////////////////////////////

class ExtractT(Operation):
    string_id = "ExtractT"

    def __init__(self, F, start, dim):
        if start + F.dim > dim or start < 0:
            raise ValueError("Index out of bound in ExtractT")
        super().__init__(F)
        self.start = start
        self.dim = dim
        self.params = (start, dim)
        self.dimarg = F.dim

    def Op(self, out, table, arg0):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        out_prev, out_mid, out_end = out.split(self.start, self.dim, self.dimarg - self.start - self.dim)
        return "\n".join(
            out_prev.assign(c_zero_float),
            VectCopy(out_mid, arg0),
            out_end.assign(c_zero_float)
        )

    def DiffT(self, v, gradin):
        f = self.children[0]
        return f.Grad(v, Extract(gradin, self.start, f.dim))

##################################################
#######                                ###########
#######     Norm related operations    ###########
#######                                ###########
##################################################


##########################
######    Norm2      #####
##########################        

def Norm2(arg):
    return Sqrt(Scalprod(arg, arg))


##########################
####    Normalize    #####
##########################        

def Normalize(arg):
    return Rsqrt(SqNorm2(arg)) * arg



##########################
######    SqDist     #####
##########################        

def SqDist(arg0, arg1):
    return SqNorm2(arg0 - arg1)


##########################
######    SqNorm2    #####
##########################        

def SqNorm2(arg0):
    return Scalprod(arg0, arg0)


#############################
######    SqNormDiag    #####
#############################

# Anisotropic (but diagonal) norm, if S.dim == A.dim:
# SqNormDiag(S,A) = sum_i s_i*a_i*a_i        

def SqNormDiag(S, A):
    return Sum(S * Square(A))


###############################################################
######     ISOTROPIC NORM : SqNormIso(S,A)    #################
###############################################################      

def SqNormIso(S, A):
    return S * SqNorm2(A)


###########################################################################
######   WEIGHTED SQUARED DISTANCE : WeightedSqDist(S,A)    ###############
###########################################################################      

def WeightedSqDist(S, A):
    return S * SqNorm2(A)


###########################################################################
####       Fully anisotropic norm, if S.dim == A.dim * A.dim          #####
###########################################################################

# SymSqNorm(A,X) = sum_{ij} a_ij * x_i*x_j
def SymSqNorm(A, X):
    return Sum(A * TensorProd(X, X))


# WeightedSqNorm(A,X) : redirects to SqNormIso, SqNormDiag or SymSqNorm
# depending on dimension of A.

def WeightedSqNorm(A, X):
    if A.dim == 1:
        return SqNormIso(A, X)
    elif A.dim == X.dim:
        return SqNormDiag(A, X)
    else:
        return SymSqNorm(A, X)


##################################################
#######                                ###########
#######     Tensor operations          ###########
#######                                ###########
##################################################


# /////////////////////////////////////////////////////////////////////////
# ////     Vector-matrix product           b x A                       ////
# /////////////////////////////////////////////////////////////////////////

class VecMatMult(Operation):
    string_id = "VecMatMult"

    def __init__(self, B, A):
        # A is vector of size n*p, interpreted as matrix, B is vector of size n, interpreted as row vector
        # output is vector of size p
        if A.dim % B.dim != 0:
            raise ValueError("Dimensions of A and B are not compatible for vector-matrix product")
        super().__init__(B, A)
        self.dim = A.dim // B.dim

    def Op(self, out, table, inB, inA):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"""
                    #if C_CONTIGUOUS     // row major
                        #pragma unroll
                        for (int i = 0; i < {self.dim}; i++) 
                        {{
                            {out.id}[i] = ({out.dtype})(0.0f);
                        	#pragma unroll
                            for (int k = 0; k < {inB.dim}; k++)
                                {out.id}[i] += {inA.id}[{self.dim} * k + i] * {inB.id}[k];
                        }}
                    #else               // column major
                        int q = 0;
                        #pragma unroll
                        for (int i = 0; i < {self.dim}; i++) 
                        {{
                            {out.id}[i] = ({out.dtype})(0.0f);
                            #pragma unroll
                            for (int k = 0; k < {inB.dim}; k++, q++)
                                {out.id}[i] += {inA.id}[q] * {inB.id}[k];
                        }}
                    #endif
                """

    def DiffT(self, v, gradin):
        B = self.children[0]
        A = self.children[1]
        return A.Grad(v, TensorProd(B, gradin)) + B.Grad(v, MatVecMult(A, gradin))


# /////////////////////////////////////////////////////////////////////////
# ////     Matrix-vector product      A x b                           ////
# /////////////////////////////////////////////////////////////////////////

class MatVecMult(Operation):
    string_id = "MatVecMult"

    def __init__(self, A, B):
        # A is vector of size n*p, interpreted as matrix, B is vector of size p, interpreted as column vector
        # output is vector of size n
        if A.dim % B.dim != 0:
            raise ValueError("Dimensions of A and B are not compatible for matrix-vector product")
        super().__init__(A, B)
        self.dim = A.dim // B.dim

    def Op(self, out, table, inB, inA):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"""
                    #if C_CONTIGUOUS     // row major
                        int q = 0;
                        #pragma unroll
                        for (int i = 0; i < {self.dim}; i++) 
                        {{
                            {out.id}[i] = ({out.dtype})(0.0f);
                            #pragma unroll
                            for (int k = 0; k < {inB.dim}; k++, q++)
                                {out.id}[i] += {inA.id}[q] * {inB.id}[k];
                        }}
                    #else               // column major
                        #pragma unroll
                        for (int i = 0; i < {self.dim}; i++) 
                        {{
                            {out.id}[i] = ({out.dtype})(0.0f);
                        	#pragma unroll
                            for (int k = 0; k < {inB.dim}; k++)
                                {out.id}[i] += {inA.id}[{self.dim} * k + i] * {inB.id}[k];
                        }}
                    #endif
                """

    def DiffT(self, v, gradin):
        A = self.children[0]
        B = self.children[1]
        return A.Grad(v, TensorProd(gradin, B)) + B.Grad(v, VecMatMult(gradin, A))


####################################
######    Tensor product       #####
####################################

class TensorProd(Operation):
    string_id = "TensorProd"

    def __init__(self, arg0, arg1):
        super().__init__(arg0, arg1)
        self.dim = arg0.dim * arg1.dim

    def Op(self, out, table, arg0, arg1):
        # returns the atomic piece of c++ code to evaluate the function on arg and return
        # the result in out
        return f"""
                    #if C_CONTIGUOUS     // row major
                        int q = 0;
        	            #pragma unroll
                        for (int k = 0; k < {arg0.dim}; k++) 
                        {{
                            #pragma unroll
                            for (int l = 0; l < {arg1.dim}; l++, q++)
                                {out.id}[q] = {arg0.id}[k] * {arg1.id}[l];
                        }}
                    #else               // column major
                        int q = 0;
                        #pragma unroll
                        for (int i = 0; i < {arg0.dim}; i++) 
                        {{
                            #pragma unroll
                            for (int j = 0; j < {arg1.dim}; j++, q++)
                                {out.id}[{arg0.dim} * j + i] = {arg0.id}[i] * {arg1.id}[j];
                        }}
                    #endif
                """

    def DiffT(self, v, gradin):
        f = self.children[0]
        g = self.children[1]
        return f.Grad(v, MatVecMult(gradin, g)) + g.Grad(v, VecMatMult(f, gradin))
