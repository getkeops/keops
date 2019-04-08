import numpy as np
from pykeops.numpy import Genred as Genred_numpy
from pykeops.numpy.utils import numpytools
try:
    import torch
    from pykeops.torch import Genred as Genred_torch
    from pykeops.torch.utils import torchtools
    usetorch = True
except ImportError:
    usetorch = False
    pass

class keops_formula:
   
    def __init__(self,x=None,axis=None):
        if x is not None:
            if isinstance(x,np.ndarray):
                self.tools = numpytools
                self.Genred = Genred_numpy
            elif usetorch and isinstance(x,torch.Tensor):
                self.tools = torchtools
                self.Genred = Genred_torch
            else:
                raise ValueError("incorrect input")
            if len(x.shape)==3:
                # init as 3d array : shape must be either (N,1,D) or (1,N,D) or (1,1,D)
                if axis is not None:
                    raise ValueError("axis should not be given for 3d array input")
                if x.shape[0]==1:
                    if x.shape[1]==1:
                        x = self.tools.view(x,(x.shape[2]))
                    else:
                        x = self.tools.view(x,(x.shape[1],x.shape[2]))
                        axis = 1
                elif x.shape[1]==1:
                    x = self.tools.view(x,(x.shape[0],x.shape[2]))
                    axis = 0
                else:
                    raise ValueError("incorrect shape for input array")
            if len(x.shape)==2:
                # init as 2d array : shape is (N,D) and axis must be given
                if axis is None:
                    raise ValueError("axis should be given")
                self.variables = (x,)
                self.dim = x.shape[1]
                # id(x) is used as temporary identifier for KeOps "Var", this will be changed when calling method "fixvariables"
                self.formula = "Var(" + str(id(x)) + "," + str(self.dim) + "," + str(axis) + ")"
                self.n = [None,None]
                self.n[axis] = x.shape[0]
                self.dtype = self.tools.dtype(x)
            elif len(x.shape)==1:
                # init as 1d array : x is a parameter
                self.variables = (x,)
                self.dim = x.shape[0]
                self.formula = "Var(" + str(id(x)) + "," + str(self.dim) + ",2)"
                self.dtype = self.tools.dtype(x)
                self.n = [None,None]
            else:
                raise ValueError("input array should be 2d or 3d")
        # N.B. we allow empty init

    def fixvariables(self):
        # we assign indices 0,1,2... id to each variable
        i = 0
        newvars = ()
        for v in self.variables:
            tag = "Var("+str(id(v))
            if tag in self.formula:
                self.formula = self.formula.replace(tag,"Var("+str(i))
                i += 1
                newvars += (v,)
        self.variables = newvars

    def joinvars(self,other):
        # we simply concatenate the two tuples of variables, without worrying about repetitions yet
        variables = self.variables + other.variables
        # now we have to check/update the two values n[0] and n[1]
        n = self.n
        for i in [0,1]:
            if n[i]:
                if other.n[i]:
                    if self.n[i]!=other.n[i]:
                        raise ValueError("incompatible sizes")
            else:
                n[i] = other.n[i]
        return variables, n


    # prototypes for operations
                    
    def unary(self,string,dim=None,opt_arg=None):
        if not dim:
            dim = self.dim
        res = keops_formula()
        res.variables, res.n = self.variables, self.n
        if opt_arg:
            res.formula = string +"(" + self.formula + "," + str(opt_arg) + ")"
        else:
            res.formula = string +"(" + self.formula + ")"
        res.dim = dim
        res.dtype = self.dtype
        res.tools = self.tools
        res.Genred = self.Genred
        return res        
                        
    def binary(self,other,string1="",string2=",",dimres=None,dimcheck="same"):
        if type(self) == type(keops_formula()):
            other = keops_formula.keopsify(other,self.tools,self.dtype)
        else:
            self = keops_formula.keopsify(self,other.tools,other.dtype)  
        if not dimres:
            dimres = self.dim
        res = keops_formula()
        res.dtype = self.dtype    
        if self.tools:
            if other.tools:
                if self.tools != other.tools:
                    raise ValueError("cannot mix numopy and torch arrays")
                else:
                    res.tools = self.tools
                    res.Genred = self.Genred
            else:
                res.tools = self.tools
                res.Genred = self.Genred
        else:
            res.tools = other.tools  
            res.Genred = other.Genred          
        if dimcheck=="same" and self.dim!=other.dim:
            raise ValueError("dimensions must be the same")
        elif dimcheck=="sameor1" and (self.dim!=other.dim and self.dim!=1 and other.dim!=1):
            raise ValueError("incorrect input dimensions")
        res.variables, res.n = self.joinvars(other)
        res.formula = string1 + "(" + self.formula + string2 + other.formula + ")"
        res.dim = dimres
        return res
        
    def keopsify(x,tools,dtype):
        if type(x) != type(keops_formula()):
            if type(x)==float:
                x = keops_formula(tools.array([x],dtype))
            elif type(x)==int:
                x = keops_formula.IntCst(x,dtype)
            else:
                raise ValueError("incorrect input")
        elif x.dtype != dtype:
            raise ValueError("data types are not compatible")
        return x       

    def IntCst(n,dtype):
        res = keops_formula()
        res.dtype = dtype
        res.variables = ()
        res.formula = "IntCst(" + str(n) + ")"
        res.dim = 1
        res.tools = None
        res.Genred = None
        res.n = [None,None]
        return res
    
    # list of operations
    
    def __add__(self,other):
        return self.binary(other,string2="+")

    def __radd__(self,other):
        if other==0:
            return self
        else:
            return keops_formula.binary(other,self,string2="+")
       
    def __sub__(self,other):
        return self.binary(other,string2="-")
        
    def __rsub__(self,other):
        if other==0:
            return -self
        else:
            return keops_formula.binary(other,self,string2="-")
        
    def __mul__(self,other):
        return self.binary(other,string2="*",dimcheck="sameor1")
        
    def __rmul__(self,other):
        if other==0:
            return O
        elif other==1:
            return self
        else:
            return keops_formula.binary(other,self,string2="*",dimcheck="sameor1")
       
    def __truediv__(self,other):
        return self.binary(other,string2="/",dimcheck="sameor1")
        
    def __rtruediv__(self,other):
        if other==0:
            return O
        else:
            return keops_formula.binary(other,self,string2="/",dimcheck="sameor1")
       
    def exp(self):
        return self.unary("Exp")
    
    def __abs__(self):
        return self.unary("Abs")
    
    def abs(self):
        return self.unary("Abs")
    
    def sqrt(self):
        return self.unary("Sqrt")
    
    def rsqrt(self):
        return self.unary("Rsqrt")
    
    def __neg__(self):
        return self.unary("Minus")
    
    def __pow__(self,n):
        if type(n)==int:
            if n==2:
                return self.unary("Square")
            else:
                return self.unary("Pow",None,n)
        elif type(n)==float:
            if n == .5:
                return self.unary("Sqrt")
            elif n == -.5:
                return self.unary("Rsqrt")
            else:
                n = keops_formula(self.tools.array([n],self.dtype))
        if type(n)==type(keops_formula()):
            if n.dim == 1 or n.dim==self.dim:
                return self.binary(n,string1="Powf",dimcheck="sameor1")
            else:
                raise ValueError("incorrect dimension of exponent")
        else:
            raise ValueError("incorrect input for exponent")

    
    # prototypes for reductions

    def unaryred(self,reduction_op,axis,dtype,opt_arg=None):
        return self.Genred(self.formula, [], reduction_op, axis, self.tools.dtypename(self.dtype), opt_arg)(*self.variables)

    def binaryred(self,other,reduction_op,axis,dtype,opt_arg=None):
        return self.Genred(self.formula, [], reduction_op, axis, self.tools.dtypename(self.dtype), opt_arg, other.formula)(*self.variables)

        
    # list of reductions

    def sum(self,axis=None,dim=None):
        if axis is None:
            axis = dim
        if axis==2:
            return self.unary("Sum",dim=1)
        else:
            self.fixvariables()    
            return self.unaryred("Sum", axis, self.dtype)
    



# convenient aliases 

def Vi(x):
    return keops_formula(x,0)
    
def Vj(x):
    return keops_formula(x,1)

def Pm(x):
    return keops_formula(x,2)
