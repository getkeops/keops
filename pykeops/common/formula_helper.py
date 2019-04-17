import numpy as np
from pykeops.numpy import Genred as Genred_numpy
from pykeops.numpy import KernelSolve as KernelSolve_numpy
from pykeops.numpy.utils import numpytools
try:
    import torch
    from pykeops.torch import Genred as Genred_torch
    from pykeops.torch import KernelSolve as KernelSolve_torch
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
    r"""
        The KeOps container class
    
        Every keops_formula object represents a mathematical formula which may depend on some "i" or "j"-indexed variables and parameter variables
        (i.e. variables that are not indexed),
        built as a combination of unary and binary operations.
    
        """   
    
    def __init__(self,x=None,axis=None):
        r"""
            Creates a KeOps variable or object
            
            :param x: may be either:
                - a float, list of floats, NumPy float, 0D or 1D NumPy array, 0D or 1D PyTorch tensor, in which case the resulting KeOps
                variable will represent a parameter variable,
                - a 2D or 3D NumPy array or PyTorch tensor, in which case the resulting KeOps variable will represent a "i"-indexed or 
                "j"-indexed variable (depending on the value of axis),
                - a tuple of 3 integers (ind,dim,cat), in which case the resulting KeOps variable will represent a symbolic variable,
                - an integer, in which case the resulting KeOps object will represent the integer.        
            :param axis: should be 0 or 1 if x is a 2D NumPy array or PyTorch tensor, should be None otherwise 
        
            Here are the behaviors of the constructor depending on the inputs x and axis:
        
            - if x is a tuple of 3 integers (ind,dim,cat), the KeOps variable will represent a symbolic variable, i.e. it will not be
              attached to an actual tensor. The corresponding tensor will need to be passed as input to the final reduction call. 
              ind correponds to the index of the variable (its position in the list of arguments in the call), dim corresponds to the
              dimension of the variable, and cat to its category ("i"-idexed if cat=0, "j"-indexed if cat=1, parameter if cat=2).
            - if x is a float, it will be converted to a single element list of floats. 
            - if x is a list of floats, the KeOps variable will represent a vector parameter (equivalent 3D shape: (1,1,len(x))). 
              The axis parameter should be None or 2. The dtype property of the variable will be set to None.
            - if x is a NumPy float (np.float32 or np.float64), it will be converted to a NumPy array of floats with same dtype.
            - if x is a NumPy 0D or 1D array or PyTorch OD or 1D tensor, the KeOps variable will represent a vector parameter (equivalent 3D shape: (1,1,len(x))).
              The dtype property of the variable will be set to the corresponding dtype ("float32" or "float64"). The axis parameter should equal 2 or None.
            - if x is a NumPy 2D array or PyTorch 2D tensor, then axis should be set to 0 or 1, and the KeOps variable will represent 
              either a "i"-indexed variable (equivalent 3D shape: (x.shape[0],1,x.shape[1])) or 
              a "j"-indexed variable (equivalent 3D shape: (1,x.shape[0],x.shape[1])).
              The dtype property of the variable will be set to the corresponding dtype ("float32" or "float64")
            - if x is a NumPy 3D array or PyTorch 3D tensor, then axis should be set to None, and the KeOps variable will represent 
              either a "i"-indexed variable if x.shape[1]=1, or a "j"-indexed variable if x.shape[0]=1 (equivalent 3D shape: x.shape).
              If x.shape[0] and x.shape[1] are greater than 1, it will result in an error.
              The dtype property of the variable will be set to the corresponding dtype ("float32" or "float64")
            - if x is an integer, the KeOps object will represent a constant integer (equivalent 3D shape: (1,1,1))
            - if x is None, the KeOps variable is initialized as empty, and the axis parameter is ignored.
              This is supposed to be used only internally.
            
            """   
        self.dtype = None
        self.variables = ()
        self.symbolic_variables = ()
        self.formula = None
        self.dim = None
        self.tools = None
        self.Genred = None
        self.KernelSolve = None
        self.ni = None
        self.nj = None
        if x is not None:
            # stage 1
            typex = type(x)
            if typex == int:
                self.formula = "IntCst(" + str(x) + ")"
                self.dim = 1
                self.axis = 2
                return
            elif typex == float:
                # convert to list and go to stage 2
                x = [x]
            elif typex == list:
                pass
            elif typex == tuple:
                # x is not a tensor but a triplet of integers (ind,dim,cat) specifying an abstract variable
                if len(x)!=3 or not isinstance(x[0],int) or not isinstance(x[1],int) or not isinstance(x[2],int):
                    raise ValueError("incorrect input")
                if axis is not None:
                    raise ValueError("axis parameter should not be given when x is of the form (ind,dim,cat)")
                self.axis = x[2]
                self.symbolic_variables = (x,)
                self.dim = x[1]
                self.formula = "VarSymb(" + str(x[0]) + "," + str(self.dim) + "," + str(self.axis) + ")"
                return
            elif typex in (np.float32,np.float64):
                x = np.array(x).reshape(1)
            elif usetorch and typex == torch.Tensor and len(x.shape)==0:
                x = x.view(1)
            # stage 2 : dealing with python list, assumed to be array of floats, treated as parameter variables without fixed dtype
            if type(x) == list:
                # init as 1d array : x is a parameter
                if axis and axis != 2:
                    raise ValueError("input is 1d vector, so it is considered as parameter and axis should equal 2")
                self.variables = (x,)
                self.dim = len(x)
                self.formula = "Var(" + str(id(x)) + "," + str(self.dim) + ",2)"
                self.axis = 2
                return
            # stage 3 : if we get here it means x must be a numpy or pytorch array
            typex = type(x)
            if typex == np.ndarray:
                self.tools = numpytools
                self.Genred = Genred_numpy
                self.KernelSolve = KernelSolve_numpy
                self.dtype = self.tools.dtype(x)
            elif usetorch and typex == torch.Tensor:
                self.tools = torchtools
                self.Genred = Genred_torch
                self.KernelSolve = KernelSolve_torch
                self.dtype = self.tools.dtype(x)
            else:
                raise ValueError("incorrect input")
            if len(x.shape)==3:
                # init as 3d array : shape must be either (N,1,D) or (1,N,D) or (1,1,D)
                # we infer axis from shape and convert to 1D or 2D array
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
            # stage 4 : now x must be 2D or 1D or 0D array
            if len(x.shape)==2:
                # init as 2d array : shape is (N,D) and axis must be given
                if axis is None:
                    raise ValueError("axis should be given")
                if axis not in (0,1):
                    raise ValueError("axis should be 0 or 1")
                # id(x) is used as temporary identifier for KeOps "Var",
                # this identifier will be changed when calling method "fixvariables"
                # But first we do a small hack, in order to distinguish same array involved twice in a formula but with 
                # different axis (e.g. Vi(x)-Vj(x) formula): we do a dummy reshape in order to get a different id
                if axis==1:
                    x = self.tools.view(x,x.shape)
                self.variables = (x,)
                self.dim = x.shape[1]
                self.formula = "Var(" + str(id(x)) + "," + str(self.dim) + "," + str(axis) + ")"
                if axis==0:
                    self.ni = x.shape[0]
                else:
                    self.nj = x.shape[0]
                self.axis = axis
                self.dtype = self.tools.dtype(x)
            elif len(x.shape)<=1:
                # init as 1d or 0d array : x is a parameter
                if axis is not None and axis != 2:
                    raise ValueError("input is 1d vector, so it is considered as parameter and axis should equal 2")
                self.variables = (x,)
                self.dim = len(x)
                self.formula = "Var(" + str(id(x)) + "," + str(self.dim) + ",2)"
                self.axis = 2
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
                    raise ValueError("incompatible " + str(prop) + ": " + str(x) + " and " + str(y))
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
        res.KernelSolve = self.KernelSolve
        res.ni = self.ni
        res.nj = self.nj
        res.variables = self.variables
        res.symbolic_variables = self.symbolic_variables
        return res
                
    def join(self,other):
        # promote props
        res = keops_formula.promote(self,other,("dtype","tools","Genred","KernelSolve","ni","nj"))
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
                        
    def binary(self,other,string,is_operator=False,dimres=None,dimcheck="sameor1"):
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
        if is_operator:
            res.formula = self.formula + string + other.formula
        else:
            res.formula = string + "(" + self.formula + "," + other.formula + ")"
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
        res.dim = self.dim
        if res.dtype is not None:
            res.fixvariables()
            res.callfun = res.Genred(res.formula, [], res.reduction_op, res.axis, res.tools.dtypename(res.dtype), res.opt_arg, res.formula2)
        if call and len(res.symbolic_variables)==0 and res.dtype is not None:
            return res()
        else:
            return res

    def kernelsolve(self,other, var=None, call=True, **kwargs):
        r"""
            Solves a positive definite linear system of the form sum(self)=other or sum(self*var)=other, using a conjugate
            gradient solver.
            
            :param self: a keops_formula object representing either a symmetric positive definite matrix 
                   or a positive definite operator. Warning!! There is no check of the symmetry and positive definiteness.
            :param other: a keops_formula variable which gives the second member of the equation.           
            :param var: either None or a symbolic keops_formula variable.
                            - If var is None then kernelsolve will return the solution var such that sum(self*var)=other.
                            - If var is a symbolic variable, it must be one of the symbolic variables contained on formula self,
                              and as a formula self must depend linearly on var. 
            :param call: either True or False. If True and if no other symbolic variable than var is contained in self,
                   then the output will be a tensor, otherwise it will be a callable keops_formula
            :param backend (string): Specifies the map-reduce scheme,
                   as detailed in the documentation of the :func:`Genred` module.
            :param device_id (int, default=-1): Specifies the GPU that should be used 
                   to perform   the computation; a negative value lets your system 
                   choose the default GPU. This parameter is only useful if your 
                   system has access to several GPUs.
            :param alpha: (float, default = 1e-10): Non-negative **ridge regularization** parameter
            :param eps:
            :params device_id: (int, default=-1): Specifies the GPU that should be used 
                   to perform   the computation; a negative value lets your system 
                   choose the default GPU. This parameter is only useful if your 
                   system has access to several GPUs.
            :param ranges: (6-uple of IntTensors, None by default):
                   Ranges of integers that specify a 
                   :doc:`block-sparse reduction scheme <../../sparsity>`
                   with *Mc clusters along axis 0* and *Nc clusters along axis 1*,
                   as detailed in the documentation 
                   of the :func:`Genred` module.

                   If **None** (default), we simply use a **dense Kernel matrix**
                   as we loop over all indices
                   :math:`i\in[0,M)` and :math:`j\in[0,N)`.
            """   

        # If given, var is symbolic variable corresponding to unknown
        # other must be a variable equal to the second member of the linear system,
        # and it may be symbolic. If it is symbolic, its index should match the index of var
        # if other is not symbolic, all variables in self must be non symbolic
        if len(other.symbolic_variables)==0 and len(self.symbolic_variables)!=0:
            raise ValueError("invalid inputs")
        
        # we infer axis of reduction as the opposite of the axis of output
        axis = 1-other.axis
        
        if var is None:
            # this is the classical mode: we want to invert sum(self*var) = other 
            # we define var as a new symbolic variable with same dimension as other
            # and we assume axis of var is same as axis of reduction
            varindex = len(self.symbolic_variables)
            var = Var(varindex,other.dim,axis)
            res = self*var
        else:
            # var is given and must be a symbolic variable which is already inside self
            varindex = var.symbolic_variables[0][0]
            res = self.init()
            res.formula = self.formula
        res.formula2 = None
        res.reduction_op = "Solve"
        res.varindex = varindex
        res.varformula = var.formula.replace("VarSymb","Var")
        res.other = other
        res.axis = axis
        res.kwargs = kwargs
        res.dim = self.dim
        if res.dtype is not None:
            res.fixvariables()
            res.callfun = res.KernelSolve(res.formula,[],res.varformula,res.axis, res.tools.dtypename(res.dtype))
        # we call if call=True, if other is not symbolic, and if the dtype is set
        if call and len(other.symbolic_variables)==0 and res.dtype is not None:
            return res()
        else:
            return res

    def __call__(self,*args, **kwargs):
        self.kwargs.update(kwargs)
        if self.dtype is None:
            if isinstance(args[0],np.ndarray):
                self.tools = numpytools
                self.Genred = Genred_numpy
                self.KernelSolve = KernelSolve_numpy
            elif usetorch and isinstance(args[0],torch.Tensor):
                self.tools = torchtools
                self.Genred = Genred_torch
                self.KernelSolve = KernelSolve_torch
            self.dtype = self.tools.dtype(args[0])
            self.fixvariables()
            if self.reduction_op == "Solve":
                self.callfun = self.KernelSolve(self.formula,[],self.formula2,self.axis, self.tools.dtypename(self.dtype))
            else:
                self.callfun = self.Genred(self.formula, [], self.reduction_op, self.axis, self.tools.dtypename(self.dtype), self.opt_arg, self.formula2)
        if self.reduction_op == "Solve" and len(self.other.symbolic_variables)==0:
            # here args should be empty, according to our rule
            if args!=():
                raise ValueError("no input required")
            # we replace by other
            args = (self.other.variables[0],)
        return self.callfun(*args, *self.variables, **self.kwargs)
    
    def __str__(self):
        string = "KeOps formula"
        tmp = self.init()
        tmp.formula = self.formula
        if hasattr(self, 'formula2'):
            tmp.formula2 = self.formula2
        else:
            tmp.formula2 = None
        tmp.fixvariables()
        string += "\n    formula: " + str(tmp.formula)
        if len(self.symbolic_variables)>0:
            string += "\n    symbolic variables: Var" + str(self.symbolic_variables[0])
            for var in self.symbolic_variables[1:]:
                string += ", Var" + str(var)
        string += "\n    shape: (" + str(self.ni) + str(",") + str(self.nj) + str(",") + str(self.dim) + ")"
        if hasattr(self, 'reduction_op'):
            string += "\n    reduction: " + self.reduction_op + " (axis=" + str(self.axis) + ")"
            if tmp.formula2 is not None:
                string += "\n        formula2: " + tmp.formula2 + "\n"
            if hasattr(self, 'opt_arg') and self.opt_arg is not None:
                string += "\n        opt_arg: " + self.opt_arg
        return string
    
    # list of operations
    
    # this flag prevents NumPy (and also PyTorch ?) to override the implementations of __radd__, __rdiv___, etc
    # that are written in this class. More concretely, if x is a NumPy array and y is a keops_formula object, writing
    # x+y will call y.__radd__(x) (keops_formula method) instead of x.__add__(y) (NumPy method)
    __array_ufunc__ = None
    
    def __add__(self,other):
        r"""
            Addition binary operation. x+y returns a new keops_formula object representing the addition of x and y.
            Broadcasting rules apply : x.dim=1 or y.dim=1 are allowed, otherwise x.dim must equal y.dim
        
            :input self: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :input other: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :returns: a keops_formula object max(self.dim,other.dim)
    
            """   
        return self.binary(other,string="+",is_operator=True)

    def __radd__(self,other):
        r"""
            Addition binary operation. x+y returns a new keops_formula object representing the addition of x and y.
            Broadcasting rules apply : x.dim=1 or y.dim=1 are allowed, otherwise x.dim must equal y.dim
        
            :input self: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :input other: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :returns: a keops_formula object max(self.dim,other.dim)
    
            """   
        if other==0:
            return self
        else:
            return keops_formula.binary(other,self,string="+",is_operator=True)
       
    def __sub__(self,other):
        r"""
            Subtraction binary operation. x-y returns a new keops_formula object representing the subtraction of x and y.
            Broadcasting rules apply : x.dim=1 or y.dim=1 are allowed, otherwise x.dim must equal y.dim
        
            :input self: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :input other: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :returns: a keops_formula object max(self.dim,other.dim)
    
            """   
        return self.binary(other,string="-",is_operator=True)
        
    def __rsub__(self,other):
        r"""
            Subtraction binary operation. x-y returns a new keops_formula object representing the subtraction of x and y.
            Broadcasting rules apply : x.dim=1 or y.dim=1 are allowed, otherwise x.dim must equal y.dim
        
            :input self: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :input other: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :returns: a keops_formula object max(self.dim,other.dim)
    
            """   
        if other==0:
            return -self
        else:
            return keops_formula.binary(other,self,string="-",is_operator=True)
        
    def __mul__(self,other):
        r"""
            Multiplication binary operation. x*y returns a new keops_formula object representing the element-wise multiplication of x and y.
            Broadcasting rules apply : x.dim=1 or y.dim=1 are allowed, otherwise x.dim must equal y.dim
        
            :input self: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :input other: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :returns: a keops_formula object max(self.dim,other.dim)
    
            """   
        return self.binary(other,string="*",is_operator=True)
        
    def __rmul__(self,other):
        r"""
            Multiplication binary operation. x*y returns a new keops_formula object representing the element-wise multiplication of x and y.
            Broadcasting rules apply : x.dim=1 or y.dim=1 are allowed, otherwise x.dim must equal y.dim
        
            :input self: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :input other: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :returns: a keops_formula object max(self.dim,other.dim)
    
            """   
        if other==0:
            return O
        elif other==1:
            return self
        else:
            return keops_formula.binary(other,self,string="*",is_operator=True)
       
    def __truediv__(self,other):
        r"""
            Division binary operation. x/y returns a new keops_formula object representing the element-wise division of x and y.
            Broadcasting rules apply : x.dim=1 or y.dim=1 are allowed, otherwise x.dim must equal y.dim
        
            :input self: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :input other: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :returns: a keops_formula object max(self.dim,other.dim)
    
            """   
        return self.binary(other,string="/",is_operator=True)
        
    def __rtruediv__(self,other):
        r"""
            Division binary operation. x/y returns a new keops_formula object representing the element-wise division of x and y.
            Broadcasting rules apply : x.dim=1 or y.dim=1 are allowed, otherwise x.dim must equal y.dim
        
            :input self: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :input other: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :returns: a keops_formula object of dimension max(self.dim,other.dim)
    
            """   
        if other==0:
            return O
        elif other==1:
            return self.unary("Inv")
        else:
            return keops_formula.binary(other,self,string="/",is_operator=True)
       
    def __or__(self,other):
        r"""
            Scalar product binary operation. x|y returns a new keops_formula object representing the euclidean scalar product of x and y.
            x.dim must equal y.dim
        
            :input self: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :input other: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :returns: a keops_formula object of dimension 1
    
            """   
        return self.binary(other,string="|",is_operator=True,dimres=1,dimcheck="same")
        
    def __ror__(self,other):
        r"""
            Scalar product binary operation. x|y returns a new keops_formula object representing the euclidean scalar product of x and y.
            x.dim must equal y.dim
        
            :input self: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :input other: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :returns: a keops_formula object of dimension 1
    
            """   
        return keops_formula.binary(other,self,string="|",is_operator=True,dimres=1,dimcheck="same")
        
    def exp(self):
        r"""
            Exponetial unary operation. exp(x) returns a new keops_formula object representing the element-wise exponential of x.
        
            :input self: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :returns: a keops_formula object of dimension self.dim
    
            """   
        return self.unary("Exp")
    
    def log(self):
        r"""
            Logarithm unary operation. log(x) returns a new keops_formula object representing the element-wise natural logarithm of x.
        
            :input self: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :returns: a keops_formula object of dimension self.dim
    
            """   
        return self.unary("Log")
    
    def sin(self):
        r"""
            Sine unary operation. sin(x) returns a new keops_formula object representing the element-wise sine function of x.
        
            :input self: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :returns: a keops_formula object of dimension self.dim
    
            """   
        return self.unary("Sin")
    
    def cos(self):
        r"""
            Cosine unary operation. cos(x) returns a new keops_formula object representing the element-wise cosine function of x.
        
            :input self: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :returns: a keops_formula object of dimension self.dim
    
            """   
        return self.unary("Cos")
    
    def __abs__(self):
        r"""
            Absolute value unary operation. abs(x) returns a new keops_formula object representing the element-wise absolute value of x.
        
            :input self: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :returns: a keops_formula object of dimension self.dim
    
            """   
        return self.unary("Abs")
    
    def abs(self):
        r"""
            Absolute value unary operation. abs(x) returns a new keops_formula object representing the element-wise absolute value of x.
        
            :input self: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :returns: a keops_formula object of dimension self.dim
    
            """   
        return abs(self)
    
    def sqrt(self):
        r"""
            Square root unary operation. sqrt(x) returns a new keops_formula object representing the element-wise square root of x.
        
            :input self: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :returns: a keops_formula object of dimension self.dim
    
            """   
        return self.unary("Sqrt")
    
    def rsqrt(self):
        r"""
            Inverse square root unary operation. rsqrt(x) returns a new keops_formula object representing the element-wise inverse square root of x.
        
            :input self: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :returns: a keops_formula object of dimension self.dim
    
            """   
        return self.unary("Rsqrt")
    
    def __neg__(self):
        r"""
            Minus unary operation. -x returns a new keops_formula object representing minus x.
        
            :input self: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :returns: a keops_formula object of dimension self.dim
    
            """   
        return self.unary("Minus")
    
    def __pow__(self,other):
        r"""
            Power binary operator. x**y returns a new keops_formula object representing x raised to the power y, element-wise.
        
            :input self: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :input other: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
                 Note:
                 - if y=2, x**y uses internally the Square KeOps operation
                 - if y=0.5, x**y uses internally the Sqrt KeOps operation
                 - if y=-0.5, x**y uses internally the Rsqrt KeOps operation
            :returns: a keops_formula object of dimension self.dim
    
            """   
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
                return self.binary(other,string="Powf",dimcheck=None)
            else:
                raise ValueError("incorrect dimension of exponent")
        else:
            raise ValueError("incorrect input for exponent")

    def power(self,other):
        r"""
            Power binary operator. power(x,y) redirects to x**y
    
            """   
        return self**other
    
    def square(self):
        r"""
            Square unary operation. square(x) returns a new keops_formula object representing the element-wise square of x.
        
            :input self: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :returns: a keops_formula object of dimension self.dim
    
            """   
        return self.unary("Square")
    
    def sign(self):
        r"""
            Sign unary operation. sign(x) returns a new keops_formula object representing the element-wise sign function applied to x.
        
            :input self: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :returns: a keops_formula object of dimension self.dim
    
            """   
        return self.unary("Sign")
    
    def step(self):
        r"""
            Step unary operation. step(x) returns a new keops_formula object representing the element-wise step function applied to x.
        
            :input self: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :returns: a keops_formula object of dimension self.dim
    
            """   
        return self.unary("Step")
    
    def relu(self):
        r"""
            ReLU unary operation. relu(x) returns a new keops_formula object representing the element-wise ReLU function applied to x.
        
            :input self: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :returns: a keops_formula object of dimension self.dim
    
            """   
        return self.unary("ReLU")
    
    def sqnorm2(self):
        r"""
            Squared norm unary operation. sqnorm2(x) returns a new keops_formula object representing the squared euclidean norm of x.
        
            :input self: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :returns: a keops_formula object of dimension 1
    
            """   
        return self.unary("SqNorm2",dimres=1)
    
    def norm2(self):
        r"""
            Norm unary operation. norm2(x) returns a new keops_formula object representing the euclidean norm of x.
        
            :input self: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :returns: a keops_formula object of dimension 1
    
            """   
        return self.unary("Norm2",dimres=1)
    
    def norm(self,dim):
        r"""
            Norm unary operation. norm(x) or norm(x,dim=2) redirects to norm2(x)
        
            :input self: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :input dim: should equal 2, otherwise an error is raised.
            :returns: a keops_formula object of dimension 1
    
            """   
        if dim != 2:
            raise ValueError("only norm over axis=2 is supported")
        return self.norm2()
    
    def normalize(self):
        r"""
            Normalize unary operation. normalize(x) returns a new keops_formula object representing x divided by its euclidean norm.
        
            :input self: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :returns: a keops_formula object of dimension self.dim
    
            """   
        return self.unary("Normalize")
    
    def sqdist(self,other):
        r"""
            Squared distance binary operator. sqdist(x,y) returns a new keops_formula object representing the euclidean squared distance between x and y.
        
            :input self: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :input other: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :returns: a keops_formula object of dimension 1
    
            """   
        return self.binary(other,string="SqDist",dimres=1)
    
    def weightedsqnorm(self,other):
        r"""
            Weighted squared norm binary operation. weightedsqnorm(s,x) returns a new keops_formula object representing a weighted squared norm of x.
            See the documentation of the corresponding WeightedSqNorm KeOps operation for explanantions.
        
            :input self: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :input other: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :returns: a keops_formula object of dimension 1
    
            """   
        if type(self) != type(keops_formula()):
            self = keops_formula.keopsify(self,other.tools,other.dtype)  
        if self.dim not in (1,other.dim,other.dim**2):
            raise ValueError("incorrect dimension of input for weightedsqnorm")
        return self.binary(other,string="WeightedSqNorm",dimres=1,dimcheck=None)
    
    def weightedsqdist(self,f,g):
        r"""
            Weighted squared distance operation. weightedsqdist(s,x,y) redirects to weightedsqnorm(s,x-y)
    
            """   
        return self.weightedsqnorm(f-g)
    
    def elem(self,i):
        r"""
            Element extraction unary operation. elem(x,i) returns a new keops_formula object representing the i-th coefficient x[i].
        
            :input self: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :input i: an integer, which should be between 0 and self.dim-1, otherwise an error is raised.
            :returns: a keops_formula object of dimension 1
    
            """   
        if type(i) is not int:
            raise ValueError("input should be integer")
        if i<0 or i>=self.dim:
            raise ValueError("index is out of bounds")
        return self.unary("Elem",dimres=1,opt_arg=i)
    
    def extract(self,i,d):
        r"""
            Range extraction unary operation. extract(x,i,d) returns a new keops_formula object representing the sub-vector x[i:i+d].
        
            :input self: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :input i: an integer, which should be between 0 and self.dim-1
            :input d: an integer, which should be between 0 and self.dim-i-1
            :returns: a keops_formula object of dimension d
    
            """   
        if (type(i) is not int) or (type(d) is not int):
            raise ValueError("inputs should be integers")
        if i<0 or i>=self.dim:
            raise ValueError("starting index is out of bounds")
        if d<1 or i+d>self.dim:
            raise ValueError("dimension is out of bounds")
        return self.unary("Extract",dimres=d,opt_arg=i,opt_arg2=d)
    
    def __getitem__(self, key):
        r"""
            Element or range extraction unary operator. x[key] redirects to elem or extract methods, depending on the key argument.
            Valid key arguments are:
                - an integer k, in which case x[key] redirects to elem(x,k),
                - a tuple :,:,k with k an integer, which is equivalent to the previous case,
                - a slice of the form k:l, k: or :l, with k and l integers, in which case x[key] redirects to extract(x,k,l-k),
                - a tuple of slices of the form :,:,k:l, :,:,k: or :,:,:l, with k and l integers, which are equivalent to the previous cases.
        
            :input self: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :input key: an integer, slice or t-uple of slices as described above.
            :returns: a keops_formula object
    
            """   
        # we allow only these forms:
        #    [:,:,k], [:,:,k:l], [:,:,k:], [:,:,:l]
        #    or equivalent [k], [k:l], [k:], [:l]
        if isinstance(key,tuple):
            if len(key)==3 and key[0]==slice(None) and key[1]==slice(None):
                key = key[2]
            else:
                raise ValueError("incorrect slicing")
        if isinstance(key,slice):
            if key.step is not None:
                raise ValueError("incorrect slicing")
            if key.start is None:
                key = slice(0,key.stop)
            if key.stop is None:
                key = slice(key.start,self.dim)
            return self.extract(key.start,key.stop-key.start)
        elif isinstance(key,int):
            return self.elem(key)
        else:
            raise ValueError("incorrect slicing")
            
    def concat(self,other):
        r"""
            Concatenation binary operation. concat(x,y) returns a new keops_formula object representing the concatenation of x and y.
        
            :input self: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :input other: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :returns: a keops_formula object of dimension self.dim + other.dim
    
            """   
        return self.binary(other,string="Concat",dimres=self.dim+other.dim,dimcheck=None)

    def concatenate(self,axis):
        r"""
            Concatenation recursive operation. concatenate(x_tuple,axis=2) returns a new keops_formula object representing the 
            concatenation of all x contained in the tuple x_tuple.
        
            :input self: a tuple of keops_formula objects, or of any inputs that can be passed to the KeOps class constructor. 
            :input axis: an integer. Should equal 2, otherwise an error is raised.
            :returns: a keops_formula object of dimension equal to the sum of the dimensions of elments in the input tuple.
    
            """   
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
        r"""
            Matrix-vector product binary operation. matvecmult(x,y) returns a new keops_formula object representing the matrix-vector product of x and y.
            See the documentation of corresponding KeOps operation MatVecMult for more explanations.
        
            :input self: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :input other: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :returns: a keops_formula object of dimension self.dim / other.dim
    
            """   
        return self.binary(other,string="MatVecMult",dimres=self.dim//other.dim,dimcheck=None)        
        
    def vecmatmult(self,other):
        r"""
            Vector-matrix product binary operation. vecmatmult(x,y) returns a new keops_formula object representing the vector-matrix product of x and y.
            See the documentation of corresponding KeOps operation VecMatMult for more explanations.
        
            :input self: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :input other: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :returns: a keops_formula object of dimension other.dim / self.dim
    
            """   
        return self.binary(other,string="VecMatMult",dimres=other.dim//self.dim,dimcheck=None)        
        
    def tensorprod(self,other):
        r"""
            Tensor product binary operation. tensorprod(x,y) returns a new keops_formula object representing the tensor product of x and y.
        
            :input self: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :input other: a keops_formula object, or any input that can be passed to the KeOps class constructor. 
            :returns: a keops_formula object of dimension self.dim * other.dim
    
            """   
        return self.binary(other,string="TensorProd",dimres=other.dim*self.dim,dimcheck=None)        
                
         




    # list of reductions

    def sum(self,axis=2,dim=None, **kwargs):
        r"""
            Summation unary operation, or Sum reduction. sum(x,axis,dim,**kwargs) will:
            - if axis=0, return the sum reduction of x over the "i" indexes.
            - if axis=1, return the sum reduction of x over the "j" indexes.
            - if axis=2, return and a new keops_formula object representing the sum of values of x,
        
            :input self: a keops_formula object, or any input that can be passed to the KeOps class constructor,
            :input axis: an integer, should be 0, 1 or 2,
            :input dim: an integer, alternative keyword for axis parameter,
            :param call: either True or False. If True and if no other symbolic variable than var is contained in self,
                   then the output will be a tensor, otherwise it will be a callable keops_formula
            :param backend (string): Specifies the map-reduce scheme,
                   as detailed in the documentation of the :func:`Genred` module.
            :param device_id (int, default=-1): Specifies the GPU that should be used 
                   to perform   the computation; a negative value lets your system 
                   choose the default GPU. This parameter is only useful if your 
                   system has access to several GPUs.
            :params device_id: (int, default=-1): Specifies the GPU that should be used 
                   to perform   the computation; a negative value lets your system 
                   choose the default GPU. This parameter is only useful if your 
                   system has access to several GPUs.
            :param ranges: (6-uple of IntTensors, None by default):
                   Ranges of integers that specify a 
                   :doc:`block-sparse reduction scheme <../../sparsity>`
                   with *Mc clusters along axis 0* and *Nc clusters along axis 1*,
                   as detailed in the documentation 
                   of the :func:`Genred` module.

                   If **None** (default), we simply use a **dense Kernel matrix**
                   as we loop over all indices
                   :math:`i\in[0,M)` and :math:`j\in[0,N)`.
            :returns: either:
                          - if axis=2: a keops_formula object of dimension 1,
                          - if axis=0 or 1, call=True and self.symbolic_variables is empty, a NumPy 2D array or PyTorch 2D tensor corresponding to the sum reduction
                            of self over axis,
                          - otherwise, a keops_formula object representing the reduction operation and which can be called as a function (see method __call__).
    
            """   
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

    
    
