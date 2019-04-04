from pykeops.numpy import Genred
import numpy as np

class keops_formula:
   
    def __init__(self,x=None,axis=None):
        if isinstance(x,np.ndarray):
            # init as 3d array : shape must be either (N,1,D) or (1,N,D) 
            if len(x.shape)==3:
                if axis is not None:
                    raise ValueError("axis should not be given for 3d array input")
                if x.shape[0]==1:
                    x = x.reshape((x.shape[1],x.shape[2]))
                    axis = 1
                elif x.shape[1]==1:
                    x = x.reshape((x.shape[0],x.shape[2]))
                    axis = 0
                else:
                    raise ValueError("incorrect shape for input array")
            # init as 2d array : shape is (N,D) and axis must be given
            if len(x.shape)==2:
                if axis is None:
                    raise ValueError("axis should be given")
                self.variables = (x,)
                self.dim = x.shape[1]
                # id(x) is used as temporary identifier for KeOps "Var", this will be changed when calling method "fixvariables"
                self.formula = "Var(" + str(id(x)) + "," + str(self.dim) + "," + str(axis) + ")"
                self.n = [None,None]
                self.n[axis] = x.shape[0]
                self.dtype = x.dtype.name
            else:
                raise ValueError("input array should be 2d or 3d")
        # N.B. we allow empty init

    def fixvariables(self):
        # we assign indices 0,1,2... id to each variable
        i = 0
        for v in self.variables:
            tag = "Var("+str(id(v))
            if tag in self.formula:
                self.formula = self.formula.replace(tag,"Var("+str(i))
                i += 1

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
        return res        
                        
    def binaryop(self,other,str,dim=None):
        if self.dtype != other.dtype:
            raise ValueError("data types are not compatible")
        if not dim:
            dim = self.dim
        res = keops_formula()
        res.variables, res.n = self.joinvars(other)
        res.formula = self.formula + str + other.formula
        res.dim = dim
        res.dtype = self.dtype
        return res


    # list of operations
    
    def __add__(self,other):
        if self.dim!=other.dim:
            raise ValueError("dimensions must be the same for add")
        return self.binaryop(other,"+")
        
    def __sub__(self,other):
        if self.dim!=other.dim:
            raise ValueError("dimensions must be the same for sub")
        return self.binaryop(other,"-")
        
    def __mul__(self,other):
        if self.dim!=other.dim:
            raise ValueError("dimensions must be the same for mul")
        return self.binaryop(other,"*")
        
    def exp(self):
        return self.unary("Exp")
    
    def __neg__(self):
        return self.unary("Minus")
    
    def __pow__(self,n):
        if n==2:
            return self.unary("Square")
        else:
            return self.unary("Pow",None,n)

    
    # prototypes for reductions

    def unaryred(self,reduction_op,axis,dtype,opt_arg=None):
        return Genred(self.formula, [], reduction_op, axis, self.dtype, opt_arg)(*self.variables)

        
    # list of reductions

    def sum(self,axis):
        self.fixvariables()    
        return self.unaryred("Sum", axis, self.dtype)
        
