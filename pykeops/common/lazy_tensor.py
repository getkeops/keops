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


# Convenient aliases:

def Var(x_or_ind,dim=None,cat=None):
    if dim is None:
        # init via data: we assume x_or_ind is data
        return LazyTensor(x_or_ind, axis=cat)
    else:
        # init via symbolic variable given as triplet (ind,dim,cat)
        return LazyTensor((x_or_ind,dim,cat))
        
def Vi(x_or_ind,dim=None):
    return Var(x_or_ind, dim, 0)    

def Vj(x_or_ind,dim=None):
    return Var(x_or_ind, dim, 1)    

def Pm(x_or_ind,dim=None):
    return Var(x_or_ind, dim, 2)    



class LazyTensor:
    r"""The KeOps container class.

    Every LazyTensor object represents a mathematical formula which 
    may depend on some "i" or "j"-indexed variables and parameter variables
    (i.e. variables that are not indexed),
    built as a combination of unary and binary operations.
    """   
    
    def __init__(self, x=None, axis=None):
        r"""Creates a KeOps variable or object
            
        Args:
            x: may be either:
        
                - a *float*, *list of floats*, *NumPy float*, *0D or 1D NumPy array*, 
                  *0D or 1D PyTorch tensor*, in which case the resulting KeOps variable will represent a parameter variable,
                - a *2D or 3D NumPy array* or *PyTorch tensor*, in which case the 
                  resulting KeOps variable will represent a "i"-indexed or 
                  "j"-indexed variable (depending on the value of axis),
                - a tuple of 3 integers (ind,dim,cat), in which case the resulting KeOps variable will represent a symbolic variable,
                - an integer, in which case the resulting KeOps object will represent the integer.                    
            axis: should be 0 or 1 if x is a 2D NumPy array or PyTorch tensor, should be None otherwise 
    
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
        self.ndim = None
        self.tools = None
        self.Genred = None
        self.KernelSolve = None
        self.ni = None
        self.nj = None

        if x is not None:  # A KeOps LazyTensor can be built from many different objects:

            # Stage 1: Are we dealing with simple numbers? ---------------------
            typex = type(x)
            
            if typex == tuple:  # x is not a Tensor but a triplet of integers 
                                # (ind,dim,cat) that specifies an abstract variable
                if len(x) != 3 or not isinstance(x[0], int) or not isinstance(x[1], int) or not isinstance(x[2], int):
                    raise ValueError("LazyTensors(tuple) is only valid if tuple = (ind,dim,cat) is a triplet of integers.")
                if axis is not None:
                    raise ValueError("'axis' parameter should not be given when 'x' is of the form (ind,dim,cat).")
                
                self.symbolic_variables = (x,)
                self.ndim = x[1]
                self.axis = x[2]
                self.formula = "VarSymb({},{},{})".format(x[0], self.ndim, self.axis)
                return  # That's it!
                
            elif typex == int:  # Integer constants are best handled directly by the compiler
                self.formula = "IntCst(" + str(x) + ")"
                self.ndim = 1
                self.axis = 2
                return  # That's it!

            elif typex == float:  # Float numbers must be encoded as Parameters,
                                  # as C++'s templating system cannot deal with floating point arithmetics.
                x = [x]  # Convert to list and go to stage 2

            elif typex == list:
                pass

            elif typex in (np.float32,np.float64):  # NumPy scalar -> NumPy array
                x = np.array(x).reshape(1)

            elif usetorch and typex == torch.Tensor and len(x.shape) == 0:  # Torch scalar -> Torch tensor
                x = x.view(1)


            # Stage 2: Dealing with python lists, understood as arrays of floats, 
            #          and handled as Parameter variables without fixed dtype
            if type(x) == list:
                if axis is not None and axis != 2:
                    raise ValueError("Lists of numbers are handled as Parameter " \
                        + "variables, with an optional 'axis' argument that is equal to 2.")
                
                self.variables = (x,)
                self.ndim = len(x)
                self.axis = 2
                self.formula = "Var({},{},2)".format(id(x), self.ndim)
                return  # That's it!


            # Stage 3: Dealing with NumPy and Torch tensors --------------------
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
                raise ValueError("LazyTensors should be built from NumPy arrays, PyTorch tensors, " \
                                 +"float/integer numbers, lists of floats or 3-uples of integers. " \
                                 +"Received: {}".format(typex))

            if len(x.shape) == 3:  # Infer axis from the input shape
                # If x is a 3D array, its shape must be either (M,1,D) or (1,N,D) or (1,1,D).
                # We infer axis from shape and convert the data to a 1D or 2D array:
                if axis is not None:
                    raise ValueError("'axis' parameter should not be given when 'x' is a 3D tensor.")

                if x.shape[0] == 1:
                    if x.shape[1] == 1:  # (1,1,D) -> Pm(D)
                        x = self.tools.view(x, (x.shape[2],) )
                    else:  # (1,N,D) -> Vj(D)
                        x = self.tools.view(x, (x.shape[1], x.shape[2]) )
                        axis = 1

                elif x.shape[1] == 1:  # (M,1,D) -> Vi(D)
                    x = self.tools.view(x, (x.shape[0], x.shape[2]) )
                    axis = 0
                else:
                    raise ValueError("If 'x' is a 3D tensor, its shape should be one of (M,1,D), (1,N,D) or (1,1,D).")


            # Stage 4: x is now encoded as a 2D or 1D array --------------------
            if len(x.shape) == 2:  # shape is (N,D) or (M,D), with an explicit 'axis' parameter
                if axis is None or axis not in (0,1):
                    raise ValueError("When 'x' is encoded as a 2D array, LazyTensor expects an explicit 'axis' value in {0,1}.")
                
                # id(x) is used as temporary identifier for KeOps "Var",
                # this identifier will be changed when calling method "fixvariables"
                # But first we do a small hack, in order to distinguish same array involved twice in a formula but with 
                # different axis (e.g. Vi(x)-Vj(x) formula): we do a dummy reshape in order to get a different id
                if axis == 1:
                    x = self.tools.view(x,x.shape)

                self.variables = (x,)
                self.ndim = x.shape[1]
                self.axis = axis
                self.formula = "Var({},{},{})".format( id(x), self.ndim, self.axis )

                if axis == 0:
                    self.ni = x.shape[0]
                else:
                    self.nj = x.shape[0]
                self.dtype = self.tools.dtype(x)

            elif len(x.shape) == 1:  # shape is (D,): x is a "Pm(D)" parameter
                if axis is not None and axis != 2:
                    raise ValueError("When 'x' is encoded as a 1D or 0D array, 'axis' must be None or 2 (= Parameter variable).")
                self.variables = (x,)
                self.ndim  = len(x)
                self.axis = 2
                self.formula = "Var({},{},2)".format( id(x), self.ndim )

            else:
                raise ValueError("LazyTensors can be built from 0D, 1D, 2D or 3D tensors. " \
                                +"Received x of shape: {}.".format(x.shape) )            
        
        # N.B.: We allow empty init!

    def fixvariables(self):
        """Assigns final labels to each variable, prior to a Genred call."""

        newvars = ()
        if self.formula2 is None: self.formula2 = ""  # We don't want to get regexp errors...

        i = len(self.symbolic_variables)  # The first few labels are already taken...
        for v in self.variables:  # So let's loop over our tensors, and give them labels:
            idv = id(v)
            if type(v) == list:
                v = self.tools.array(v, self.dtype)

            # Replace "Var(idv," by "Var(i," and increment 'i':
            tag = "Var({},".format(idv)
            if tag in self.formula + self.formula2:
                self.formula  = self.formula.replace( tag, "Var({},".format(i))
                self.formula2 = self.formula2.replace(tag, "Var({},".format(i))
                i += 1
                newvars += (v,)

        # "VarSymb(..)" appear when users rely on the "LazyTensor(Ind,Dim,Cat)" syntax,
        # for the sake of disambiguation:
        self.formula  = self.formula.replace( "VarSymb(", "Var(")  # We can now replace them with
        self.formula2 = self.formula2.replace("VarSymb(", "Var(")  # actual "Var" symbols

        if self.formula2 == "": self.formula2 = None  # The pre-processing step is now over
        self.variables = newvars

    def promote(self, other, props):
        """Creates a new LazyTensor whose 'None' properties are set to those of 'self' or 'other'."""
        res = LazyTensor()

        for prop in props:
            x, y = getattr(self, prop), getattr(other, prop)
            if x is not None:
                if y is not None and x != y:
                    raise ValueError("Incompatible values for attribute {}: {} and {}.".format(prop, x, y))
                setattr(res, prop, x)
            else:
                setattr(res, prop, y)
        return res
        
    def init(self):
        """Creates a copy of a LazyTensor, without 'formula'."""
        res = LazyTensor()
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
        """Merges the variables and attributes of two LazyTensors, with a compatibility check.
        
        N.B.: This method concatenates tuples of variables, without paying attention to repetitions.
        """
        res = LazyTensor.promote(self, other, ("dtype","tools","Genred","KernelSolve","ni","nj") )
        
        res.variables = self.variables + other.variables
        res.symbolic_variables = self.symbolic_variables + other.symbolic_variables
        return res


    # Prototypes for unary and binary operations  ==============================
                    
    def unary(self, operation, dimres=None, opt_arg=None, opt_arg2=None):
        """Symbolically applies 'operation' to self, with optional arguments if needed.
        
        The optional argument 'dimres' may be used to specify the dimension of the output 'result'.
        """
        if not dimres: dimres = self.ndim

        res = self.init()  # Copy of self, without a formula
        if opt_arg2 is not None:
            res.formula = "{}({},{},{})".format(operation, self.formula, opt_arg, opt_arg2) 
        elif opt_arg is not None:
            res.formula = "{}({},{})".format(operation, self.formula, opt_arg) 
        else:
            res.formula = "{}({})".format(operation, self.formula) 
        res.ndim = dimres
        return res        
                        
    def binary(self, other, operation, is_operator=False, dimres=None, dimcheck="sameor1"):
        """Symbolically applies 'operation' to self, with optional arguments if needed.
        
        Keyword args:
          - dimres (int or None): may be used to specify the dimension of the output 'result'.
          - is_operator (True or False): may be used to specify if **operation** is
            an operator like "+", "-" or a 'genuine' function.
          - dimcheck (string): shall we check the input dimensions?
            Supported values are "same" and "sameor1".
        """
        # If needed, convert float numbers / lists / arrays / tensors to LazyTensors:
        if not isinstance(self,LazyTensor):  self = LazyTensor(self)
        if not isinstance(other,LazyTensor): other = LazyTensor(other)      

        # By default, the dimension of the output variable is the max of the two operands:
        if not dimres: dimres = max(self.ndim, other.ndim)

        if dimcheck == "same" and self.ndim != other.ndim:
            raise ValueError("Operation {} expects inputs of the same dimension. " \
                           + "Received {} and {}.".format(operation, self.ndim, other.ndim))

        elif dimcheck == "sameor1" and (self.ndim != other.ndim and self.ndim != 1 and other.ndim != 1):
            raise ValueError("Operation {} expects inputs of the same dimension or dimension 1. " \
                           + "Received {} and {}.".format(operation, self.ndim, other.ndim))

        res = LazyTensor.join(self, other)  # Merge the attributes and variables of both operands
        res.ndim = dimres
        
        if is_operator:
            res.formula = "({} {} {})".format( self.formula, operation, other.formula )
        else:
            res.formula = "{}({}, {})".format( operation, self.formula, other.formula )
        
        return res
        

    # Prototypes for reduction operations  =====================================

    def reduction(self, reduction_op, other=None, opt_arg=None, axis=None, dim=None, call=True, **kwargs):
        """Applies a reduction to a LazyTensor.

        Keyword Args:
          - other: 
          - opt_arg: typically, some integer needed by ArgKMin reductions.
          - axis or dim (0 or 1): the axis with respect to which the reduction should be performed.
          - call (True or False): Should we actually perform the reduction on the current variables?
            If True, the returned object will be a NumPy array or a PyTorch tensor.
            Otherwise, we simply return a callable LazyTensor that can be used
            as a :mod:`pykeops.numpy.Genred` or :mod:`pykeops.torch.Genred` function 
            on arbitrary input data.
        """

        if axis is None:  axis = dim  # NumPy uses axis, PyTorch uses dim...
        if axis not in (0,1): raise ValueError("Reductions must be called with 'axis' (or 'dim') equal to 0 or 1.")
        
        if other is None:
            res = self.init()  # ~ self.copy()
            res.formula2 = None
        else:
            res = self.join(other)
            res.formula2 = other.formula

        res.formula = self.formula
        res.reduction_op = reduction_op
        res.axis = axis
        res.opt_arg = opt_arg
        res.kwargs = kwargs
        res.ndim = self.ndim

        if res.dtype is not None:
            res.fixvariables()  # Turn the "id(x)" numbers into consecutive labels
            # "res" now becomes a callable object:
            res.callfun = res.Genred(res.formula, [], res.reduction_op, res.axis, 
                                     res.tools.dtypename(res.dtype), res.opt_arg, res.formula2)
        
        if call and len(res.symbolic_variables) == 0 and res.dtype is not None:
            return res()
        else:
            return res

    def kernelsolve(self, other, var=None, call=True, **kwargs):
        r"""Solves a positive definite linear system of the form sum(self)=other or sum(self*var)=other, using a conjugate
            gradient solver.
            
        :param self: a LazyTensor object representing either a symmetric positive definite matrix 
                or a positive definite operator. Warning!! There is no check of the symmetry and positive definiteness.
        :param other: a LazyTensor variable which gives the second member of the equation.           
        :param var: either None or a symbolic LazyTensor variable.
                        - If var is None then kernelsolve will return the solution var such that sum(self*var)=other.
                        - If var is a symbolic variable, it must be one of the symbolic variables contained on formula self,
                            and as a formula self must depend linearly on var. 
        :param call: either True or False. If True and if no other symbolic variable than var is contained in self,
                then the output will be a tensor, otherwise it will be a callable LazyTensor
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
        if len(other.symbolic_variables) == 0 and len(self.symbolic_variables) != 0:
            raise ValueError("If 'self' has symbolic variables, so should 'other'.")
        
        # we infer axis of reduction as the opposite of the axis of output
        axis = 1 - other.axis
        
        if var is None:
            # this is the classical mode: we want to invert sum(self*var) = other 
            # we define var as a new symbolic variable with same dimension as other
            # and we assume axis of var is same as axis of reduction
            varindex = len(self.symbolic_variables)
            var = Var(varindex, other.ndim, axis)
            res = self * var
        else:
            # var is given and must be a symbolic variable which is already inside self
            varindex = var.symbolic_variables[0][0]
            res = self.init()
            res.formula = self.formula

        res.formula2 = None
        res.reduction_op = "Solve"
        res.varindex = varindex
        res.varformula = var.formula.replace("VarSymb", "Var")
        res.other = other
        res.axis = axis
        res.kwargs = kwargs
        res.ndim = self.ndim

        if res.dtype is not None:
            res.fixvariables()
            res.callfun = res.KernelSolve(res.formula, [], res.varformula, 
                                          res.axis, res.tools.dtypename(res.dtype))
        
        # we call if call=True, if other is not symbolic, and if the dtype is set
        if call and len(other.symbolic_variables) == 0 and res.dtype is not None:
            return res()
        else:
            return res

    def __call__(self, *args, **kwargs):
        """Executes a Genred or KernelSolve call on input data, as specified by self.formula."""
        self.kwargs.update(kwargs)

        if self.dtype is None:  # This can only happen if we haven't encountered 2D or 3D arrays just yet...
            if isinstance(args[0], np.ndarray):  # We use the "NumPy" or "PyTorch" backend depending on the first argument
                self.tools = numpytools
                self.Genred = Genred_numpy
                self.KernelSolve = KernelSolve_numpy

            elif usetorch and isinstance(args[0], torch.Tensor):
                self.tools = torchtools
                self.Genred = Genred_torch
                self.KernelSolve = KernelSolve_torch
                
            self.dtype = self.tools.dtype(args[0])
            self.fixvariables()
            if self.reduction_op == "Solve":
                self.callfun = self.KernelSolve(self.formula, [], self.formula2,
                                                self.axis, self.tools.dtypename(self.dtype))
            else:
                self.callfun = self.Genred(self.formula, [], self.reduction_op, 
                                    self.axis, self.tools.dtypename(self.dtype), self.opt_arg, self.formula2)
        
        if self.reduction_op == "Solve" and len(self.other.symbolic_variables) == 0:
            # here args should be empty, according to our rule
            if args != ():
                raise ValueError("no input required")
            # we replace by other
            args = (self.other.variables[0],)
        return self.callfun(*args, *self.variables, **self.kwargs)
    
    def __str__(self):
        """Returns a verbose string identifier."""
        tmp = self.init()  # ~ self.copy()
        tmp.formula = self.formula
        tmp.formula2 = None if not hasattr(self, 'formula2') else self.formula2
        tmp.fixvariables()  # Replace Var(id(x),...) with consecutive labels

        string = "KeOps LazyTensor\n    formula: {}".format(tmp.formula)
        if len(self.symbolic_variables) > 0:
            string += "\n    symbolic variables: Var{}".format(self.symbolic_variables[0])
            for var in self.symbolic_variables[1:]:  string += ", Var{}".format(var)

        string += "\n    shape: {}".format(self.shape)

        if hasattr(self, 'reduction_op'):
            string += "\n    reduction: {} (axis={})".format(self.reduction_op, self.axis)
            if tmp.formula2 is not None:
                string += "\n        formula2: {}".format(tmp.formula2)
            if hasattr(self, 'opt_arg') and self.opt_arg is not None:
                string += "\n        opt_arg: {}".format(self.opt_arg)
        return string
    
    @property
    def shape(self):
        ni   = 1 if self.ni   is None else self.ni
        nj   = 1 if self.nj   is None else self.nj
        ndim = 1 if self.ndim is None else self.ndim
        return (ni, nj, ndim)

    def dim(self):
        """Just as in PyTorch, returns the number of dimensions of a LazyTensor."""
        return len(self.shape)

    # List of supported operations  ============================================
    
    # N.B.: This flag prevents NumPy (and also PyTorch ?) from overriding 
    #       the KeOps implementations of __radd__, __rdiv___, etc. written below. 
    #       For instance, if x is a NumPy array and y is a KeOps LazyTensor, 
    #       writing  "x+y"  will call y.__radd__(x) (LazyTensor method) instead 
    #       of x.__add__(y) (NumPy method)
    __array_ufunc__ = None
    
    # Arithmetics --------------------------------------------------------------
    def __add__(self,other):
        r"""Broadcasted addition operator - a binary operation. 
        
        ``x + y`` returns a :mod:`LazyTensor` that encodes, 
        symbolically, the addition of ``x`` and ``y``.
        """   
        return self.binary(other, "+", is_operator=True)

    def __radd__(self,other):
        r"""Broadcasted addition operator - a binary operation. 
        
        ``x + y`` returns a :mod:`LazyTensor` that encodes, 
        symbolically, the addition of ``x`` and ``y``.
        """   
        if other == 0:
            return self
        else:
            return LazyTensor.binary(other, self, "+", is_operator=True)
       
    def __sub__(self,other):
        r"""Broadcasted subtraction operator - a binary operation. 
        
        ``x - y`` returns a :mod:`LazyTensor` that encodes, 
        symbolically, the subtraction of ``x`` and ``y``.
        """
        return self.binary(other, "-", is_operator=True)
        
    def __rsub__(self,other):
        r"""Broadcasted subtraction operator - a binary operation. 
        
        ``x - y`` returns a :mod:`LazyTensor` that encodes, 
        symbolically, the subtraction of ``x`` and ``y``.
        """
        if other == 0:
            return -self
        else:
            return LazyTensor.binary(other, self, "-", is_operator=True)
        
    def __mul__(self,other):
        r"""Broadcasted elementwise product - a binary operation. 
        
        ``x * y`` returns a :mod:`LazyTensor` that encodes, symbolically, 
        the elementwise product of ``x`` and ``y``.
        """
        return self.binary(other, "*", is_operator=True)
        
    def __rmul__(self,other):
        r"""Broadcasted elementwise product - a binary operation. 
        
        ``x * y`` returns a :mod:`LazyTensor` that encodes, symbolically, 
        the elementwise product of ``x`` and ``y``.
        """ 
        if   other == 0: return 0
        elif other == 1: return self
        else: return LazyTensor.binary(other, self, "*", is_operator=True)
       
    def __truediv__(self,other):
        r"""Broadcasted elementwise division - a binary operation. 
        
        ``x / y`` returns a :mod:`LazyTensor` that encodes, symbolically, 
        the elementwise division of ``x`` by ``y``.
        """  
        return self.binary(other, "/", is_operator=True)
        
    def __rtruediv__(self,other):
        r"""Broadcasted elementwise division - a binary operation. 
        
        ``x / y`` returns a :mod:`LazyTensor` that encodes, symbolically, 
        the elementwise division of ``x`` by ``y``.
        """  
        if   other == 0: return 0
        elif other == 1: return self.unary("Inv")
        else: return LazyTensor.binary(other, self, "/", is_operator=True)
       
    def __or__(self, other):
        r"""Euclidean scalar product - a binary operation.

        ``(x|y)`` returns a :mod:`LazyTensor` that encodes, symbolically, 
        the scalar product of ``x`` and ``y`` which are assumed to have the same shape.    
        """   
        return self.binary(other, "|", is_operator=True, dimres=1, dimcheck="same")
        
    def __ror__(self,other):
        r"""Euclidean scalar product - a binary operation.

        ``(x|y)`` returns a :mod:`LazyTensor` that encodes, symbolically, 
        the scalar product of ``x`` and ``y`` which are assumed to have the same shape.    
        """
        return LazyTensor.binary(other, self, "|", is_operator=True, dimres=1, dimcheck="same")

    # Unary arithmetics --------------------------------------------------------

    def __abs__(self):
        r"""Element-wise absolute value - a unary operation.
        
        ``abs(x)`` returns a :mod:`LazyTensor` that encodes, symbolically, 
        the element-wise absolute value of ``x``.
        """  
        return self.unary("Abs")
    
    def abs(self):
        r"""Element-wise absolute value - a unary operation.
        
        ``x.abs()`` returns a :mod:`LazyTensor` that encodes, symbolically, 
        the element-wise absolute value of ``x``.
        """  
        return abs(self)
    
    def __neg__(self):
        r"""Element-wise minus - a unary operation.
        
        ``-x`` returns a :mod:`LazyTensor` that encodes, symbolically, 
        the element-wise opposite of ``x``.
        """  
        return self.unary("Minus")
    

    # Simple functions ---------------------------------------------------------
       
    def exp(self):
        r"""Element-wise exponential - a unary operation.
        
        ``x.exp()`` returns a :mod:`LazyTensor` that encodes, symbolically, 
        the element-wise exponential of ``x``.
        """   
        return self.unary("Exp")
    
    def log(self):
        r"""Element-wise logarithm - a unary operation.
        
        ``x.log()`` returns a :mod:`LazyTensor` that encodes, symbolically, 
        the element-wise logarithm of ``x``.
        """    
        return self.unary("Log")
    
    def cos(self):
        r"""Element-wise cosine - a unary operation.
        
        ``x.cos()`` returns a :mod:`LazyTensor` that encodes, symbolically, 
        the element-wise cosine of ``x``.
        """  
        return self.unary("Cos")

    def sin(self):
        r"""Element-wise sine - a unary operation.
        
        ``x.sin()`` returns a :mod:`LazyTensor` that encodes, symbolically, 
        the element-wise sine of ``x``.
        """     
        return self.unary("Sin")
    
    def sqrt(self):
        r"""Element-wise square root - a unary operation.
        
        ``x.sqrt()`` returns a :mod:`LazyTensor` that encodes, symbolically, 
        the element-wise square root of ``x``.
        """    
        return self.unary("Sqrt")
    
    def rsqrt(self):
        r"""Element-wise inverse square root - a unary operation.
        
        ``x.rsqrt()`` returns a :mod:`LazyTensor` that encodes, symbolically, 
        the element-wise inverse square root of ``x``.
        """   
        return self.unary("Rsqrt")

    def __pow__(self, other):
        r"""Broadcasted element-wise power operator - a binary operation.
        
        ``x**y`` returns a :mod:`LazyTensor` that encodes, symbolically, 
        the element-wise value of ``x`` to the power ``y``.
        
        Note:
          - if **y = 2**, ``x**y`` relies on the ``"Square"`` KeOps operation;
          - if **y = 0.5**, ``x**y`` uses on the ``"Sqrt"`` KeOps operation;
          - if **y = -0.5**, ``x**y`` uses on the ``"Rsqrt"`` KeOps operation.
        """   
        if type(other) == int:
            return self.unary("Square") if other == 2 else self.unary("Pow", opt_arg=other)
        
        elif type(other) == float:
            if   other ==  .5: return self.unary("Sqrt")
            elif other == -.5: return self.unary("Rsqrt")
            else: other = LazyTensor(self.tools.array([other], self.dtype))

        if type(other) == type(LazyTensor()):
            if other.ndim == 1 or other.ndim == self.ndim:
                return self.binary(other, "Powf", dimcheck=None)
            else:
                raise ValueError("Incompatible dimensions for the LazyTensor and its exponent: " \
                               + "{} and {}.".format(self.ndim, other.ndim))
        else:
            raise ValueError("The exponent should be an integer, a floar number or a LazyTensor.")

    def power(self,other):
        r"""Broadcasted element-wise power operator - a binary operation.
        
        ``pow(x,y)`` is equivalent to ``x**y``.
        """   
        return self**other
    
    def square(self):
        r"""Element-wise square - a unary operation.
        
        ``x.square()`` is equivalent to ``x**2`` and returns a :mod:`LazyTensor` 
        that encodes, symbolically, the element-wise square of ``x``.
        """
        return self.unary("Square")
    
    def sign(self):
        r"""Element-wise sign in {-1,0,+1} - a unary operation.
        
        ``x.sign()`` returns a :mod:`LazyTensor` that encodes, symbolically, 
        the element-wise sign of ``x``.
        """  
        return self.unary("Sign")
    
    def step(self):
        r"""Element-wise step function - a unary operation.
        
        ``x.step()`` returns a :mod:`LazyTensor` that encodes, symbolically, 
        the element-wise sign of ``x``.
        """   
        return self.unary("Step")
    
    def relu(self):
        r"""Element-wise ReLU function - a unary operation.
        
        ``x.relu()`` returns a :mod:`LazyTensor` that encodes, symbolically,
        the element-wise positive part of ``x``.
        """   
        return self.unary("ReLU")
    
    def sqnorm2(self):
        r"""Squared Euclidean norm - a unary operation.
        
        ``x.sqnorm2()`` returns a :mod:`LazyTensor` that encodes, symbolically, 
        the squared Euclidean norm of a vector ``x``.
        """
        return self.unary("SqNorm2", dimres=1)
    
    def norm2(self):
        r"""Euclidean norm - a unary operation.
        
        ``x.norm2()`` returns a :mod:`LazyTensor` that encodes, symbolically, 
        the Euclidean norm of a vector ``x``.
        """  
        return self.unary("Norm2",dimres=1)
    
    def norm(self, dim):
        r"""Euclidean norm - a unary operation.

        ``x.norm(-1)`` is equivalent to ``x.norm2()`` and returns a 
        :mod:`LazyTensor` that encodes, symbolically, the Euclidean norm of a vector ``x``.
        """
        if dim not in [-1, 2]:
            raise ValueError("KeOps only supports norms over the last dimension.")
        return self.norm2()
    
    def normalize(self):
        r"""Vector normalization - a unary operation.
        
        ``x.normalize()`` returns a :mod:`LazyTensor` that encodes, symbolically, 
        a vector ``x`` divided by its Euclidean norm.
        """   
        return self.unary("Normalize")
    
    def sqdist(self,other):
        r"""Squared distance - a binary operation.
        
        ``x.sqdist(y)`` returns a :mod:`LazyTensor` that encodes, symbolically,
        the squared Euclidean distance between two vectors ``x`` and ``y``.
        """   
        return self.binary(other, "SqDist", dimres=1)
    
    def weightedsqnorm(self, other):
        r"""Weighted squared norm - a binary operation. 
        
        ``LazyTensor.weightedsqnorm(s, x)`` returns a :mod:`LazyTensor` that encodes, symbolically,
        the weighted squared Norm of a vector ``x`` - see
        the :doc:`main reference page <../api/math-operations>` for details.
        """   
        if type(self) != type(LazyTensor()):
            self = LazyTensor(self)  
        
        if self.ndim not in (1, other.ndim, other.ndim**2):
            raise ValueError("Squared norm weights should be of size 1 (scalar), " \
                            +"D (diagonal) or D^2 (full symmetric tensor), but received " \
                            +"{} with D={}.".format(self.ndim, other.ndim))
        
        return self.binary(other, "WeightedSqNorm", dimres=1, dimcheck=None)
    
    def weightedsqdist(self, f, g):
        r"""Weighted squared distance. 
        
        ``LazyTensor.weightedsqdist(s, x, y)`` is equivalent to ``LazyTensor.weightedsqnorm(s, x - y)``.
        """   
        return self.weightedsqnorm( f - g )
    
    def elem(self, i):
        r"""Indexing of a vector - a unary operation. 
        
        ``x.elem(i)`` returns a :mod:`LazyTensor` that encodes, symbolically,
        the i-th element ``x[i]`` of the vector ``x``.
        """   
        if type(i) is not int:
            raise ValueError("Elem indexing is only supported for integer indices.")
        if i < 0 or i >= self.ndim:
            raise ValueError("Index i={} is out of bounds [0,D) = [0,{}).".format(i, self.ndim))
        return self.unary("Elem", dimres=1, opt_arg=i)
    
    def extract(self, i, d):
        r"""Range indexing - a unary operation.
        
        ``x.extract(i, d)`` returns a :mod:`LazyTensor` that encodes, symbolically,
        the sub-vector ``x[i:i+d]`` of the vector ``x``.    
        """   
        if (type(i) is not int) or (type(d) is not int):
            raise ValueError("Indexing is only supported for integer indices.")
        if i < 0 or i >= self.ndim:
            raise ValueError("Starting index is out of bounds.")
        if d < 1 or i+d > self.ndim:
            raise ValueError("Slice dimension is out of bounds.")
        return self.unary("Extract", dimres=d, opt_arg=i, opt_arg2=d)
    
    def __getitem__(self, key):
        r"""Element or range indexing - a unary operation.

        ``x[key]`` redirects to the :meth:`elem` or :meth:`extract` methods, depending on the ``key`` argument.
        Supported values are:

            - an integer ``k``, in which case ``x[key]`` 
              redirects to ``elem(x,k)``,
            - a tuple ``:,:,k`` with ``k`` an integer, 
              which is equivalent to the case above,
            - a slice of the form ``k:l``, ``k:`` or ``:l``, with ``k`` 
              and ``l`` two integers, in which case ``x[key]`` redirects to ``extract(x,k,l-k)``,
            - a tuple of slices of the form ``:,:,k:l``, ``:,:,k:`` or ``:,:,:l``, 
              with ``k`` and ``l`` two integers, which are equivalent to the case above.
        """   
        # we allow only these forms:
        #    [:,:,k], [:,:,k:l], [:,:,k:], [:,:,:l]
        #    or equivalent [k], [k:l], [k:], [:l]
        if isinstance(key, tuple):
            if len(key) == 3 and key[0] == slice(None) and key[1] == slice(None):
                key = key[2]
            else:
                raise ValueError("LazyTensors only support indexing with respect to their last dimension.")
        
        if isinstance(key, slice):
            if not key.step in [None, 1]:
                raise ValueError("LazyTensors do not support sliced indexing with stepsizes > 1.")
            if key.start is None:
                key = slice(0, key.stop)
            if key.stop is None:
                key = slice(key.start, self.ndim)
            return self.extract(key.start, key.stop - key.start)
        elif isinstance(key, int):
            return self.elem(key)
        else:
            raise ValueError("LazyTensors only support indexing with integers and vanilla python slices.")
            
    def concat(self,other):
        r"""Concatenation of two LazyTensors - a binary operation.
        
        ``x.concat(y)`` returns a :mod:`LazyTensor` that encodes, symbolically,
        the concatenation of ``x`` and ``y`` along their last dimension.    
        """   
        return self.binary(other, "Concat", dimres = (self.ndim + other.ndim), dimcheck=None)

    def concatenate(self, axis):
        r"""Concatenation of a tuple of LazyTensors.
        
        ``LazyTensor.concatenate( (x_1, x_2, ..., x_n), -1)`` returns a :mod:`LazyTensor` that encodes, symbolically,
        the concatenation of ``x_1``, ``x_2``, ..., ``x_n`` along their last dimension.    
        Note that **axis** should be equal to -1 or 2 (if the ``x_i``'s are 3D LazyTensors):
        LazyTensors only support concatenation and indexing operations with respect
        to the last dimension.
        """   
        if axis not in [-1, 2]:
            raise ValueError("LazyTensor only supports concatenation along the last axis.")
        if isinstance(self, tuple):
            if len(self) == 0:
                raise ValueError("Received an empty tuple of LazyTensors.")
            elif len(self) == 1:
                return self
            elif len(self) == 2:    
                return self[0].concat(self[1])
            else:
                return LazyTensor.concatenate(self[0].concat(self[1]), self[2:], axis=2)
        else:
            raise ValueError("LazyTensor.concatenate is implemented on *tuples* of LazyTensors.")    
    
    def cat(self, dim):
        r"""Concatenation of a tuple of LazyTensors.
        
        ``LazyTensor.cat( (x_1, x_2, ..., x_n), -1)`` 
        is a PyTorch-friendly alias for ``LazyTensor.concatenate( (x_1, x_2, ..., x_n), -1)``;
        just like indexing operations, it is only supported along the last dimension.
        """
        return LazyTensor.concatenate(self, dim)

    def matvecmult(self,other):
        r"""Matrix-vector product - a binary operation.

        If ``x.shape[-1] == A*B`` and ``y.shape[-1] == B``,
        ``z = x.matvecmult(y)`` returns a :mod:`LazyTensor` 
        such that ``z.shape[-1] == A`` which encodes, symbolically,
        the matrix-vector product of ``x`` and ``y`` along their last dimension.
        For details, please check the documentation of the KeOps operation ``"MatVecMult"`` in
        the :doc:`main reference page <../api/math-operations>`.    
        """
        return self.binary(other, "MatVecMult", dimres = (self.ndim // other.ndim), dimcheck=None)        
        
    def vecmatmult(self,other):
        r"""Vector-matrix product - a binary operation.

        If ``x.shape[-1] == A`` and ``y.shape[-1] == A*B``,
        ``z = x.vecmatmult(y)`` returns a :mod:`LazyTensor` 
        such that ``z.shape[-1] == B`` which encodes, symbolically,
        the vector-matrix product of ``x`` and ``y`` along their last dimension.
        For details, please check the documentation of the KeOps operation ``"VetMacMult"`` in
        the :doc:`main reference page <../api/math-operations>`.    
        """
        return self.binary(other, "VecMatMult", dimres = (other.ndim // self.ndim), dimcheck=None)        
        
    def tensorprod(self,other):
        r"""Tensor product of vectors - a binary operation.

        If ``x.shape[-1] == A`` and ``y.shape[-1] == B``,
        ``z = x.tensorprod(y)`` returns a :mod:`LazyTensor` 
        such that ``z.shape[-1] == A*B`` which encodes, symbolically,
        the tensor product of ``x`` and ``y`` along their last dimension.
        For details, please check the documentation of the KeOps operation ``"TensorProd"`` in
        the :doc:`main reference page <../api/math-operations>`.    
        """ 
        return self.binary(other, "TensorProd", dimres = (other.ndim * self.ndim), dimcheck=None)        
                
         




    # List of supported reductions  ============================================

    def sum(self, axis=2, dim=None, **kwargs):
        r"""Summation unary operation, or Sum reduction. 
        
        sum(x, axis, dim, **kwargs) will:
        
          - if **axis or dim = 0**, return the sum reduction of x over the "i" indexes.
          - if **axis or dim = 0**, return the sum reduction of x over the "j" indexes.
          - if **axis or dim = 0**, return and a new LazyTensor object representing the sum of values of x,
        
        :input self: a LazyTensor object, or any input that can be passed to the KeOps class constructor,
        :input axis: an integer, should be 0, 1 or 2,
        :input dim: an integer, alternative keyword for axis parameter,
        :param call: either True or False. If True and if no other symbolic variable than var is contained in self,
                then the output will be a tensor, otherwise it will be a callable LazyTensor
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
                        - if axis=2: a LazyTensor object of dimension 1,
                        - if axis=0 or 1, call=True and self.symbolic_variables is empty, a NumPy 2D array or PyTorch 2D tensor corresponding to the sum reduction
                        of self over axis,
                        - otherwise, a LazyTensor object representing the reduction operation and which can be called as a function (see method __call__).
        """   
        if dim is not None:
            axis = dim
        if axis in [-1, 2]:
            return self.unary("Sum", dimres=1)
        else:
            return self.reduction("Sum", axis=axis, **kwargs)
    
    def sum_reduction(self,**kwargs):
        return self.reduction("Sum", **kwargs)
    
    def logsumexp(self, weight=None, **kwargs):
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
        return self.reduction("KMin", opt_arg=K, **kwargs)

    def Kmin_reduction(self, **kwargs):
        return self.Kmin(**kwargs)

    def argKmin(self, K, **kwargs):
        return self.reduction("ArgKMin", opt_arg=K, **kwargs)

    def argKmin_reduction(self, **kwargs):
        return self.argKmin(**kwargs)

    def Kmin_argKmin(self, K, **kwargs):
        return self.reduction("KMin_ArgKMin", opt_arg=K, **kwargs)

    def Kmin_argKmin_reduction(self, **kwargs):
        return self.Kmin_argKmin(**kwargs)

    
    
