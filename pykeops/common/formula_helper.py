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


# convenient aliases 

def Var(x_or_ind,dim=None,cat=None):
    if dim is None:
        # init via data: we assume x_or_ind is data
        return keops_formula(x_or_ind,axis=cat)
    else:
        # init via symbolic variable given as triplet (ind,dim,cat)
        return keops_formula((x_or_ind,dim,cat))
        
def Vi(x_or_ind,dim=None):
    return Var(x_or_ind,dim,0)    

def Vj(x_or_ind,dim=None):
    return Var(x_or_ind,dim,1)    

def Pm(x_or_ind,dim=None):
    return Var(x_or_ind,dim,2)    



class keops_formula:
   
    def __init__(self,x=None,axis=None):
        self.dtype = None
        self.variables = ()
        self.symbolic_variables = ()
        self.formula = None
        self.dim = None
        self.tools = None
        self.Genred = None
        self.ni = None
        self.nj = None
        self.dtype = None
        if x is not None:
            typex = type(x)
            if typex == int:
                self.formula = "IntCst(" + str(x) + ")"
                self.dim = 1
                return
            elif typex == float:
                x = [x]
            elif typex == list:
                pass
            elif typex == tuple:
                # x is not a tensor but a triplet of integers (ind,dim,cat) specifying an abstract variable
                if len(x)!=3 or not isinstance(x[0],int) or not isinstance(x[1],int) or not isinstance(x[2],int):
                    raise ValueError("incorrect input")
                if axis is not None:
                    raise ValueError("axis parameter should not be given when x is of the form (ind,dim,cat)")
                axis = x[2]
                self.symbolic_variables = (x,)
                self.dim = x[1]
                self.formula = "VarSymb(" + str(x[0]) + "," + str(self.dim) + "," + str(axis) + ")"
                return
            elif typex == np.ndarray:
                self.tools = numpytools
                self.Genred = Genred_numpy
                self.dtype = self.tools.dtype(x)
            elif usetorch and typex == torch.Tensor:
                self.tools = torchtools
                self.Genred = Genred_torch
                self.dtype = self.tools.dtype(x)
            else:
                raise ValueError("incorrect input")
            if type(x) == list or len(x.shape)==1:
                # init as 1d array : x is a parameter
                if axis and axis != 2:
                    raise ValueError("input is 1d vector, so it is considered as parameter and axis should equal 2")
                self.variables = (x,)
                self.dim = len(x)
                self.formula = "Var(" + str(id(x)) + "," + str(self.dim) + ",2)"
                return
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
                if axis not in (0,1):
                    raise ValueError("axis should be 0 or 1")
                self.variables = (x,)
                self.dim = x.shape[1]
                # id(x) is used as temporary identifier for KeOps "Var", this will be changed when calling method "fixvariables"
                self.formula = "Var(" + str(id(x)) + "," + str(self.dim) + "," + str(axis) + ")"
                if axis==0:
                    self.ni = x.shape[0]
                else:
                    self.nj = x.shape[0]
                self.dtype = self.tools.dtype(x)
            else:
                raise ValueError("input array should be 1d, 2d or 3d")            
        # N.B. we allow empty init

    def fixvariables(self):
        # we assign final indices to each variable
        i = len(self.symbolic_variables)
        newvars = ()
        if self.formula2 is None:
            self.formula2 = ""
        for v in self.variables:
            idv = id(v)
            if type(v) == list:
                v = self.tools.array(v,self.dtype)
            tag = "Var("+str(idv)
            if tag in self.formula + self.formula2:
                self.formula = self.formula.replace(tag,"Var("+str(i))
                self.formula2 = self.formula2.replace(tag,"Var("+str(i))
                i += 1
                newvars += (v,)
        self.formula = self.formula.replace("VarSymb(","Var(")
        self.formula2 = self.formula2.replace("VarSymb(","Var(")
        if self.formula2 == "":
            self.formula2 = None
        self.variables = newvars

    def promote(self,other,props):
        res = keops_formula()
        for prop in props:
            x = getattr(self,prop)
            y = getattr(other,prop)
            if x is not None:
                if y is not None and x!=y:
                    raise ValueError("incompatible ",prop,": ",x," and ",y)
                setattr(res,prop,x)
            else:
                setattr(res,prop,y)
        return res
        
    def init(self):
        # create new object and propagate properties found in self
        res = keops_formula()
        res.dtype = self.dtype
        res.tools = self.tools
        res.dtype = self.dtype
        res.Genred = self.Genred
        res.ni = self.ni
        res.nj = self.nj
        res.variables = self.variables
        res.symbolic_variables = self.symbolic_variables
        return res
                
    def join(self,other):
        # promote props
        res = keops_formula.promote(self,other,("dtype","tools","Genred","ni","nj"))
        # we simply concatenate the two tuples of variables, without worrying about repetitions yet
        res.variables = self.variables + other.variables
        res.symbolic_variables = self.symbolic_variables + other.symbolic_variables
        return res


    # prototypes for operations
                    
    def unary(self,string,dimres=None,opt_arg=None,opt_arg2=None):
        if not dimres:
            dimres = self.dim
        res = self.init()
        if opt_arg2 is not None:
            res.formula = string +"(" + self.formula + "," + str(opt_arg) + "," + str(opt_arg2) + ")"
        elif opt_arg is not None:
            res.formula = string +"(" + self.formula + "," + str(opt_arg) + ")"
        else:
            res.formula = string +"(" + self.formula + ")"
        res.dim = dimres
        return res        
                        
    def binary(self,other,string1="",string2=",",dimres=None,dimcheck="sameor1"):
        if not isinstance(self,keops_formula):
            self = keops_formula(self)
        if not isinstance(other,keops_formula):
            other = keops_formula(other)      
        if not dimres:
            dimres = max(self.dim,other.dim)
        if dimcheck=="same" and self.dim!=other.dim:
            raise ValueError("dimensions must be the same")
        elif dimcheck=="sameor1" and (self.dim!=other.dim and self.dim!=1 and other.dim!=1):
            raise ValueError("incorrect input dimensions")
        res = keops_formula.join(self,other)
        res.formula = string1 + "(" + self.formula + string2 + other.formula + ")"
        res.dim = dimres
        return res
        

    # prototypes for reductions

    def reduction(self,reduction_op,other=None,opt_arg=None,axis=None, dim=None, call=True, **kwargs):
        if axis is None:
            axis = dim
        if axis not in (0,1):
            raise ValueError("axis must be 0 or 1 for reduction")
        if other is None:
            res = self.init()
            res.formula2 = None
        else:
            res = self.join(other)
            res.formula2 = other.formula
        res.formula = self.formula
        res.reduction_op = reduction_op
        res.axis = axis
        res.opt_arg = opt_arg
        res.kwargs = kwargs
        if res.dtype is not None:
            res.fixvariables()
            res.redop = res.Genred(res.formula, [], res.reduction_op, res.axis, res.tools.dtypename(res.dtype), res.opt_arg, res.formula2)
        if call and len(self.symbolic_variables)==0 and res.dtype is not None:
            return res()
        else:
            return res

    def __call__(self,*args, **kwargs):
        self.kwargs.update(kwargs)
        if self.dtype is None:
            if isinstance(args[0],np.ndarray):
                self.tools = numpytools
                self.Genred = Genred_numpy
            elif usetorch and isinstance(args[0],torch.Tensor):
                self.tools = torchtools
                self.Genred = Genred_torch
            self.dtype = self.tools.dtype(args[0])
            self.fixvariables()
            self.redop = Genred(self.formula, [], self.reduction_op, self.axis, self.tools.dtypename(res.dtype), self.opt_arg, self.formula2)
        return self.redop(*args, *self.variables, **self.kwargs)

            
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
        return self.binary(other,string2="*")
        
    def __rmul__(self,other):
        if other==0:
            return O
        elif other==1:
            return self
        else:
            return keops_formula.binary(other,self,string2="*")
       
    def __truediv__(self,other):
        return self.binary(other,string2="/")
        
    def __rtruediv__(self,other):
        if other==0:
            return O
        elif other==1:
            return self.unary("Inv")
        else:
            return keops_formula.binary(other,self,string2="/")
       
    def __or__(self,other):
        return self.binary(other,string2="|",dimres=1,dimcheck="same")
        
    def __ror__(self,other):
        return keops_formula.binary(other,self,string2="|",dimres=1,dimcheck="same")
        
    def exp(self):
        return self.unary("Exp")
    
    def log(self):
        return self.unary("Log")
    
    def sin(self):
        return self.unary("Sin")
    
    def cos(self):
        return self.unary("Cos")
    
    def __abs__(self):
        return self.unary("Abs")
    
    def abs(self):
        return abs(self)
    
    def sqrt(self):
        return self.unary("Sqrt")
    
    def rsqrt(self):
        return self.unary("Rsqrt")
    
    def __neg__(self):
        return self.unary("Minus")
    
    def __pow__(self,other):
        if type(other)==int:
            if other==2:
                return self.unary("Square")
            else:
                return self.unary("Pow",opt_arg=other)
        elif type(other)==float:
            if other == .5:
                return self.unary("Sqrt")
            elif other == -.5:
                return self.unary("Rsqrt")
            else:
                other = keops_formula(self.tools.array([other],self.dtype))
        if type(other)==type(keops_formula()):
            if other.dim == 1 or other.dim==self.dim:
                return self.binary(other,string1="Powf",dimcheck=None)
            else:
                raise ValueError("incorrect dimension of exponent")
        else:
            raise ValueError("incorrect input for exponent")

    def power(self,other):
        return self**other
    
    def square(self):
        return self.unary("Square")
    
    def sqrt(self):
        return self.unary("Sqrt")
    
    def rsqrt(self):
        return self.unary("Rsqrt")
    
    def sign(self):
        return self.unary("Sign")
    
    def step(self):
        return self.unary("Step")
    
    def relu(self):
        return self.unary("ReLU")
    
    def sqnorm2(self):
        return self.unary("SqNorm2",dimres=1)
    
    def norm2(self):
        return self.unary("Norm2",dimres=1)
    
    def norm(self,dim):
        if dim != 2:
            raise ValueError("only norm over axis=2 is supported")
        return self.norm2()
    
    def normalize(self):
        return self.unary("Normalize")
    
    def sqdist(self,other):
        return self.binary(other,string1="SqDist",dimres=1)
    
    def weightedsqnorm(self,other):
        if type(self) != type(keops_formula()):
            self = keops_formula.keopsify(self,other.tools,other.dtype)  
        if self.dim not in (1,other.dim,other.dim**2):
            raise ValueError("incorrect dimension of input for weightedsqnorm")
        return self.binary(other,string1="WeightedSqNorm",dimres=1,dimcheck=None)
    
    def weightedsqdist(self,f,g):
        return self.weightedsqnorm(f-g)
    
    def elem(self,i):
        if type(i) is not int:
            raise ValueError("input should be integer")
        if i<0 or i>=self.dim:
            raise ValueError("index is out of bounds")
        return self.unary("Elem",dimres=1,opt_arg=i)
    
    def extract(self,i,d):
        if (type(i) is not int) or (type(d) is not int):
            raise ValueError("inputs should be integers")
        if i<0 or i>=self.dim:
            raise ValueError("starting index is out of bounds")
        if d<0 or i+d>=self.dim:
            raise ValueError("dimension is out of bounds")
        return self.unary("Extract",dimres=d,opt_arg=i,opt_arg2=d)
    
    def __getitem__(self, key):
        if self.ni is None and self.nj is None:
            if not(isinstance(key,int) or isinstance(key,slice)):
                raise ValueError("incorrect slicing")
        else:
            if not isinstance(key,tuple) or len(key)!=3 or key[0]!=slice(None) or key[1]!=slice(None):
                raise ValueError("incorrect slicing")
            key = key[2]
        if isinstance(key,slice):
            if key.step is not None:
                raise ValueError("incorrect slicing")
            if key.start is None:
                key.start = 0
            if key.stop is None:
                key.stop = self.dim
            return self.extract(key.start,key.stop-key.start)
        elif isinstance(key,int):
            return self.elem(key)
            
    def concat(self,other):
        return self.binary(other,string1="Concat",dimres=self.dim+other.dim,dimcheck=None)

    def concatenate(self,axis):
        if axis != 2:
            raise ValueError("only concatenation over axis=2 is supported")
        if isinstance(self,tuple):
            if len(self)==0:
                raise ValueError("tuple must not be empty")
            elif len(self)==1:
                return self
            elif len(self)==2:    
                return self[0].concat(self[1])
            else:
                return keops_formula.concatenate(self[0].concat(self[1]),self[2:],axis=2)
        else:
            raise ValueError("input must be tuple")    
    
    def matvecmult(self,other):
        return self.binary(other,string1="MatVecMult",dimres=self.dim//other.dim,dimcheck=None)        
        
    def vecmatmult(self,other):
        return self.binary(other,string1="VecMatMult",dimres=other.dim//self.dim,dimcheck=None)        
        
    def tensorprod(self,other):
        return self.binary(other,string1="TensorProd",dimres=other.dim*self.dim,dimcheck=None)        
                
         




    # list of reductions

    def sum(self,axis=2,dim=None, **kwargs):
        if dim is not None:
            axis = dim
        if axis==2:
            return self.unary("Sum",dimres=1)
        else:
            return self.reduction("Sum", axis=axis, **kwargs)
    
    def sum_reduction(self,**kwargs):
        return self.reduction("Sum",**kwargs)
    
    def logsumexp(self,weight=None,**kwargs):
        if weight is None:
            return self.reduction("LogSumExp", **kwargs)
        else:
            return self.reduction("LogSumExpWeight", weight, **kwargs)
        
    def logsumexp_reduction(self,**kwargs):
        return self.logsumexp(**kwargs)
        
    def sumsoftmaxweight(self,weight,**kwargs):
        return self.reduction("SumSoftMaxWeight", weight, **kwargs)
        
    def sumsoftmaxweight_reduction(self,**kwargs):
        return self.sumsoftmaxweight(**kwargs)
        
    def min(self, **kwargs):
        return self.reduction("Min", **kwargs)
    
    def min_reduction(self,**kwargs):
        return self.min(**kwargs)

    def __min__(self, **kwargs):
        return self.min(**kwargs)

    def argmin(self, **kwargs):
        return self.reduction("ArgMin", **kwargs)
    
    def argmin_reduction(self,weight,**kwargs):
        return self.argmin(**kwargs)

    def min_argmin(self, **kwargs):
        return self.reduction("Min_ArgMin", **kwargs)
    
    def min_argmin_reduction(self, **kwargs):
        return self.min_argmin(**kwargs)
    
    def max(self, **kwargs):
        return self.reduction("Max", **kwargs)
    
    def max_reduction(self,weight,**kwargs):
        return self.max(**kwargs)

    def __max__(self, **kwargs):
        return self.max(**kwargs)

    def argmax(self, **kwargs):
        return self.reduction("ArgMax", **kwargs)
    
    def argmax_reduction(self,weight,**kwargs):
        return self.argmax(**kwargs)

    def max_argmax(self, **kwargs):
        return self.reduction("Max_ArgMax", **kwargs)
    
    def max_argmax_reduction(self, **kwargs):
        return self.max_argmax(**kwargs)
    
    def Kmin(self, K, **kwargs):
        return self.reduction("KMin",opt_arg=K,**kwargs)

    def Kmin_reduction(self, **kwargs):
        return self.Kmin(**kwargs)

    def argKmin(self, K, **kwargs):
        return self.reduction("ArgKMin",opt_arg=K,**kwargs)

    def argKmin_reduction(self, **kwargs):
        return self.argKmin(**kwargs)

    def Kmin_argKmin(self, K, **kwargs):
        return self.reduction("KMin_ArgKMin",opt_arg=K,**kwargs)

    def Kmin_argKmin_reduction(self, **kwargs):
        return self.Kmin_argKmin(**kwargs)

    
    



