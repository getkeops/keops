import copy
import re

import math

import numpy as np
from pykeops.common.utils import check_broadcasting


def same_or_one_test(*dims):
    # test wether input dimensions are compatible with broadcasting
    return len(set(list(dims) + [1])) <= 2


def is_scalar_and_equals(x, val):
    # test wether the input x is a Python scalar and
    # that its value equals val
    if isinstance(x, (int, float, complex)) and not isinstance(x, bool):
        return x == val
    else:
        return False


def is_complex_lazytensor(x):
    return isinstance(x, ComplexGenericLazyTensor)


class GenericLazyTensor:
    r"""Symbolic wrapper for NumPy arrays and PyTorch tensors. This is the abstract class,
    end user should use :class:`pykeops.numpy.LazyTensor` or :class:`pykeops.torch.LazyTensor`.

    :class:`LazyTensor` encode numerical arrays through the combination
    of a symbolic, **mathematical formula** and a list of **small data arrays**.
    They can be used to implement efficient algorithms on objects
    that are **easy to define**, but **impossible to store** in memory
    (e.g. the matrix of pairwise distances between
    two large point clouds).

    :class:`LazyTensor` may be created from standard NumPy arrays or PyTorch tensors,
    combined using simple mathematical operations and converted
    back to NumPy arrays or PyTorch tensors with
    efficient reduction routines, which outperform
    standard tensorized implementations by two orders of magnitude.
    """

    variables = ()
    symbolic_variables = ()
    formula = None
    formula2 = None
    ndim = None
    tools = None
    Genred = None
    KernelSolve = None
    batchdims = None
    ni = None
    nj = None
    axis = None
    ranges = None  # Block-sparsity pattern
    backend = None  # "CPU", "GPU", "GPU_2D", etc.
    _dtype = None
    is_complex = False

    def __init__(self, x=None, axis=None):
        r"""Creates a KeOps symbolic variable.

        Args:
            x: May be either:

                - A *float*, a *list of floats*, a *NumPy float*, a *0D or 1D NumPy array*,
                  a *0D or 1D PyTorch tensor*, in which case the :class:`LazyTensor`
                  represents a constant **vector of parameters**, to be broadcasted
                  on other :class:`LazyTensor`.
                - A *2D NumPy array* or *PyTorch tensor*, in which case the :class:`LazyTensor`
                  represents a **variable** indexed by :math:`i` if **axis=0** or :math:`j` if **axis=1**.
                - A *3D+ NumPy array* or *PyTorch tensor* with a dummy dimension (=1) at position -3 or -2,
                  in which case the :class:`LazyTensor` represents a **variable** indexed by
                  :math:`i` or :math:`j`, respectively. Dimensions before the last three will be handled as
                  **batch dimensions**, that may support operator broadcasting.
                - A *tuple of 3 integers* (ind,dim,cat), in which case the
                  :class:`LazyTensor` represents a :doc:`symbolic variable <../../../api/math-operations>`
                  that should be instantiated at call-time.
                - An *integer*, in which case the :class:`LazyTensor` represents an **integer constant** handled
                  efficiently at compilation time.
                - **None**, for internal use.

            axis (int): should be equal to 0 or 1 if **x** is a 2D tensor, and  **None** otherwise.

        .. warning::

            A :class:`LazyTensor` constructed
            from a NumPy array or a PyTorch tensor retains its **dtype** (float16, float32 or float64)
            and **device** properties (is it stored on the GPU?).
            Since KeOps does **not** support automatic type conversions and data transfers,
            please make sure **not to mix** :class:`LazyTensor`
            that come from different frameworks/devices or which are stored
            with different precisions.
        """

        # Duck typing attribute, to be used instead of "isinstance(self, GenericLazyTensor)"
        # This is a workaround for an importlib.reload messy situation,
        # and will come handy when we'll start supporting Block GenericLazyTensors
        # and other custom operators.
        self.__GenericLazyTensor__ = True

        self.get_tools()

        # A KeOps LazyTensor can be built from many different objects:
        if x is not None:

            # Stage 1: Are we dealing with simple numbers? ---------------------
            typex = type(x)

            if typex == tuple:  # x is not a Tensor but a triplet of integers
                # (ind,dim,cat) that specifies an abstract variable
                if (
                    len(x) != 3
                    or not isinstance(x[0], int)
                    or not isinstance(x[1], int)
                    or not isinstance(x[2], int)
                ):
                    raise ValueError(
                        "LazyTensors(tuple) is only valid if tuple = (ind,dim,cat) is a triplet of integers."
                    )
                if axis is not None:
                    raise ValueError(
                        "'axis' parameter should not be given when 'x' is of the form (ind,dim,cat)."
                    )

                self.symbolic_variables = (x,)
                self.ndim = x[1]
                self.axis = x[2]
                self.formula = "VarSymb({},{},{})".format(x[0], self.ndim, self.axis)
                return  # That's it!

            # Integer constants are best handled directly by the compiler
            elif typex == int:
                self.formula = "IntCst(" + str(x) + ")"
                self.ndim = 1
                self.axis = 2
                return  # That's it!

            # Float numbers must be encoded as Parameters,  as C++'s templating system cannot deal
            # with floating point arithmetics.
            elif typex in self.tools.float_types:
                x = [x]  # Convert to list and go to stage 2
                typex = list

            # Stage 2: Dealing with python lists, understood as arrays of floats,
            #          and handled as Parameter variables without fixed dtype
            if typex == list:
                if axis is not None and axis != 2:
                    raise ValueError(
                        "Lists of numbers are handled as Parameter "
                        "variables, with an optional 'axis' argument that is equal to 2."
                    )

                self.variables = (x,)
                self.ndim = len(x)
                self.axis = 2
                self.formula = "Var({},{},2)".format(id(x), self.ndim)
                return  # That's it!
            else:
                self._dtype = self.tools.dtypename(self.tools.dtype(x))

        typex = type(x)

        if (
            typex
            not in [type(None), tuple, int, float, list, self.tools.arraytype]
            + self.tools.float_types
        ):
            raise TypeError(
                "LazyTensors should be built from " + self.tools.arrayname + ", "
                "float/integer numbers, lists of floats or 3-uples of integers. "
                "Received: {}".format(typex)
            )

        if typex == self.tools.arraytype and len(x.shape) == 0:
            x = x.view(1)
        elif typex in self.tools.float_types:
            x = self.tools.arraytype([x]).view(1)

        if typex == self.tools.arraytype:
            if len(x.shape) >= 3:  # Infer axis from the input shape
                # If x is a 3D+ array, its shape must be either (..,M,1,D) or (..,1,N,D) or (..,1,1,D).
                # We infer axis from shape and squeeze out the "1" dimensions:
                if axis is not None:
                    raise ValueError(
                        "'axis' parameter should not be given when 'x' is a 3D tensor."
                    )

                if len(x.shape) > 3:  # We're in "batch mode"
                    self.batchdims = tuple(x.shape[:-3])

                if x.shape[-3] == 1:
                    if x.shape[-2] == 1:  # (..,1,1,D) -> Pm(D)
                        x = x.squeeze(-2).squeeze(-2)
                        axis = 2
                    else:  # (..,1,N,D) -> Vj(D)
                        x = x.squeeze(-3)
                        axis = 1

                elif x.shape[-2] == 1:  # (M,1,D) -> Vi(D)
                    x = x.squeeze(-2)
                    axis = 0
                else:
                    raise ValueError(
                        "If 'x' is a 3D+ tensor, its shape should be one of (..,M,1,D), (..,1,N,D) or (..,1,1,D)."
                    )

            # Stage 4: x is now encoded as a 2D or 1D array + batch dimensions --------------------
            if (
                len(x.shape) >= 2 and axis != 2
            ):  # shape is (..,M,D) or (..,N,D), with an explicit 'axis' parameter
                if axis is None or axis not in (0, 1):
                    raise ValueError(
                        "When 'x' is encoded as a 2D array, LazyTensor expects an explicit 'axis' value in {0,1}."
                    )

                # id(x) is used as temporary identifier for KeOps "Var",
                # this identifier will be changed when calling method "fixvariables"
                # But first we do a small hack, in order to distinguish same array involved twice in a formula but with
                # different axis (e.g. Vi(x)-Vj(x) formula): we do a dummy reshape in order to get a different id
                if axis == 1:
                    x = self.tools.view(x, x.shape)

                self.variables = (x,)
                self.ndim = x.shape[-1]
                self.axis = axis
                self.formula = "Var({},{},{})".format(id(x), self.ndim, self.axis)

                if axis == 0:
                    self.ni = x.shape[-2]
                else:
                    self.nj = x.shape[-2]

                self._dtype = self.tools.dtypename(self.tools.dtype(x))

            elif (
                len(x.shape) == 1 or axis == 2
            ):  # shape is (D,): x is a "Pm(D)" parameter
                if axis is not None and axis != 2:
                    raise ValueError(
                        "When 'x' is encoded as a 1D or 0D array, 'axis' must be None or 2 (= Parameter variable)."
                    )
                self.variables = (x,)
                self.ndim = x.shape[-1]
                self.axis = 2
                self.formula = "Var({},{},2)".format(id(x), self.ndim)

            else:
                raise ValueError(
                    "LazyTensors can be built from 0D, 1D, 2D or 3D+ tensors. "
                    + "Received x of shape: {}.".format(x.shape)
                )

    def lt_constructor(self, x=None, axis=None):
        r"""This method is specialized in :class:`pykeops.numpy.LazyTensor` and :class:`pykeops.torch.LazyTensor`. It
        returns a new instance of a LazyTensor (numpy or pytorch)."""
        pass

    def get_tools(self):
        r"""This method is specialized in :class:`pykeops.numpy.LazyTensor` and :class:`pykeops.torch.LazyTensor`. It
        populates the tools class."""
        pass

    def fixvariables(self):
        r"""If needed, assigns final labels to each variable and pads their batch dimensions prior to a :mod:`Genred()` call."""
        newvars = ()
        if self.formula2 is None:
            self.formula2 = ""  # We don't want to get regexp errors...
        device = None  # Useful to load lists (and float constants) on the proper device
        for v in self.variables:
            device = self.tools.device(v)
            if device is not None:
                break
        i = len(self.symbolic_variables)  # The first few labels are already taken...

        # So let's loop over our tensors, and give them labels:
        for v in self.variables:
            idv = id(v)
            if type(v) == list:
                v = self.tools.array(v, self._dtype, device)

            # Replace "Var(idv," by "Var(i," and increment 'i':
            tag = "Var({},".format(idv)
            if tag in self.formula + self.formula2:
                self.formula = self.formula.replace(tag, "Var({},".format(i))
                self.formula2 = self.formula2.replace(tag, "Var({},".format(i))
                # Detect if v is meant to be used as a variable or as a parameter:
                str_cat_v = re.search(
                    r"Var\({},\d+,([012])\)".format(i), self.formula + self.formula2
                ).group(1)
                is_variable = 1 if str_cat_v in ("0", "1") else 0
                dims_to_pad = self.nbatchdims + 1 + is_variable - len(v.shape)
                padded_v = self.tools.view(v, (1,) * dims_to_pad + v.shape)
                newvars += (padded_v,)
                if (
                    hasattr(self, "rec_multVar_highdim")
                    and self.rec_multVar_highdim == idv
                ):
                    self.rec_multVar_highdim = i
                i += 1

        # "VarSymb(..)" appear when users rely on the "LazyTensor(Ind,Dim,Cat)" syntax,
        # for the sake of disambiguation:
        self.formula = self.formula.replace(
            "VarSymb(", "Var("
        )  # We can now replace them with
        self.formula2 = self.formula2.replace(
            "VarSymb(", "Var("
        )  # actual "Var" symbols
        if self.formula2 == "":
            self.formula2 = None  # The pre-processing step is now over
        self.variables = newvars

    def separate_kwargs(self, kwargs):
        # separating keyword arguments for Genred init vs Genred call...
        # Currently the only four additional optional keyword arguments that are passed to Genred init
        # are accuracy options: dtype_acc, use_double_acc and sum_scheme,
        # chunk mode option enable_chunks,
        # and compiler option optional_flags.
        kwargs_init = []
        kwargs_call = []
        for key in kwargs:
            if key in (
                "dtype_acc",
                "use_double_acc",
                "sum_scheme",
                "enable_chunks",
                "optional_flags",
            ):
                kwargs_init += [(key, kwargs[key])]
            else:
                kwargs_call += [(key, kwargs[key])]
        kwargs_init = dict(kwargs_init)
        kwargs_call = dict(kwargs_call)
        return kwargs_init, kwargs_call

    def promote(self, other, props, is_complex=False):
        r"""
        Creates a new :class:`LazyTensor` whose **None** properties are set to those of **self** or **other**.
        """
        res = self.lt_constructor(is_complex=is_complex)

        for prop in props:
            y, x = getattr(self, prop), getattr(other, prop)
            if x is not None:
                if y is not None:
                    if prop == "ranges":
                        x_eq_y = all(
                            tuple(
                                map(lambda x, y: self.tools.eq(x, y).all().item(), x, y)
                            )
                        )
                    else:
                        x_eq_y = x == y
                    if not (x_eq_y):
                        raise ValueError(
                            "Incompatible values for attribute {}: {} and {}.".format(
                                prop, x, y
                            )
                        )
                setattr(res, prop, x)
            else:
                setattr(res, prop, y)
        return res

    def init(self, is_complex=False):
        r"""
        Creates a copy of a :class:`LazyTensor`, without **formula** attribute.
        """
        res = self.lt_constructor(is_complex=is_complex)
        res.tools = self.tools
        res._dtype = self._dtype
        res.Genred = self.Genred
        res.KernelSolve = self.KernelSolve
        res.batchdims = self.batchdims
        res.ni = self.ni
        res.nj = self.nj
        res.ranges = self.ranges
        res.backend = self.backend
        res.variables = self.variables
        res.symbolic_variables = self.symbolic_variables
        return res

    def join(self, other, is_complex=False):
        r"""
        Merges the variables and attributes of two :class:`LazyTensor`, with a compatibility check.
        This method concatenates tuples of variables, without paying attention to repetitions.
        """
        res = self.promote(
            other,
            (
                "_dtype",
                "tools",
                "Genred",
                "KernelSolve",
                "ni",
                "nj",
                "ranges",
                "backend",
            ),
            is_complex=is_complex,
        )
        res.symbolic_variables = self.symbolic_variables + other.symbolic_variables

        # Checks on the batch dimensions - we support broadcasting:
        res.batchdims = check_broadcasting(self.batchdims, other.batchdims)
        # N.B.: If needed, variables will be padded with "dummy 1's" just before the Genred call, in self/res.fixvariables():
        res.variables = self.variables + other.variables

        return res

    # Prototypes for unary and binary operations  ==============================

    def unary(
        self, operation, dimres=None, opt_arg=None, opt_arg2=None, is_complex=None
    ):
        r"""
        Symbolically applies **operation** to **self**, with optional arguments if needed.

        The optional argument **dimres** may be used to specify the dimension of the output **result**.
        """

        if is_complex is None:
            is_complex = self.is_complex

        # we must prevent any operation if self is the output of a reduction operation,
        # i.e. if it has a reduction_op field
        if hasattr(self, "reduction_op"):
            raise ValueError(
                "Input is a 'reduced' LazyTensor, no operation can be applied to it. "
            )

        if not dimres:
            dimres = self.ndim

        res = self.init(is_complex)  # Copy of self, without a formula
        if opt_arg2 is not None:
            res.formula = "{}({},{},{})".format(
                operation, self.formula, opt_arg, opt_arg2
            )
        elif opt_arg is not None:
            res.formula = "{}({},{})".format(operation, self.formula, opt_arg)
        else:
            res.formula = "{}({})".format(operation, self.formula)
        res.ndim = dimres
        return res

    def binary(
        self,
        other,
        operation,
        is_operator=False,
        dimres=None,
        dimcheck="sameor1",
        opt_arg=None,
        opt_pos="last",
        rversion=False,
        is_complex=None,
    ):
        r"""Symbolically applies **operation** to **self**, with optional arguments if needed.

        Keyword args:
          - dimres (int): May be used to specify the dimension of the output **result**.
          - is_operator (bool, default=False): May be used to specify if **operation** is
            an operator like ``+``, ``-`` or a "genuine" function.
          - dimcheck (string): shall we check the input dimensions?
            Supported values are ``"same"``, ``"sameor1"``, or **None**.
          - rversion (Boolean): shall we invert lhs and rhs of the binary op, e.g. as in __radd__, __rmut__, etc...
        """

        # If needed, convert float numbers / lists / arrays / tensors to LazyTensors:
        if not hasattr(other, "__GenericLazyTensor__"):
            other = self.lt_constructor(other)

        if is_complex is None:
            is_complex = True if (self.is_complex or other.is_complex) else False

        # we must prevent any operation if self or other is the output of a reduction operation,
        # i.e. if it has a reduction_op field
        if hasattr(self, "reduction_op") or hasattr(other, "reduction_op"):
            raise ValueError(
                "One of the inputs is a 'reduced' LazyTensor, no operation can be applied to it. "
            )

        # By default, the dimension of the output variable is the max of the two operands:
        if not dimres:
            dimres = max(self.ndim, other.ndim)

        if dimcheck == "same":
            if self.ndim != other.ndim:
                raise ValueError(
                    "Operation {} expects inputs of the same dimension. ".format(
                        operation
                    )
                    + "Received {} and {}.".format(self.ndim, other.ndim)
                )

        elif dimcheck == "sameor1":
            if self.ndim != other.ndim and self.ndim != 1 and other.ndim != 1:
                raise ValueError(
                    "Operation {} expects inputs of the same dimension or dimension 1. ".format(
                        operation
                    )
                    + "Received {} and {}.".format(self.ndim, other.ndim)
                )

        elif dimcheck != None:
            raise ValueError("incorrect dimcheck keyword in binary operation")

        res = self.join(
            other, is_complex=is_complex
        )  # Merge the attributes and variables of both operands

        res.ndim = dimres

        if not rversion:
            lformula, rformula = self.formula, other.formula
        else:
            rformula, lformula = self.formula, other.formula

        if is_operator:
            res.formula = "({} {} {})".format(lformula, operation, rformula)
        elif opt_arg is not None:
            if hasattr(opt_arg, "__GenericLazyTensor__"):
                opt_arg = opt_arg.formula
            if opt_pos == "last":
                res.formula = "{}({}, {}, {})".format(
                    operation, lformula, rformula, opt_arg
                )
            elif opt_pos == "middle":
                res.formula = "{}({}, {}, {})".format(
                    operation, lformula, opt_arg, rformula
                )
        else:
            res.formula = "{}({}, {})".format(operation, lformula, rformula)

        # special case of multiplication with a variable V : we define a special tag to enable factorization in case
        # the user requires a sum reduction over the opposite index (or any index if V is a parameter):
        # for example sum_i V_j k(x_i,y_j) = V_j sum_i k(x_i,y_j), so we will use KeOps reduction for the kernel
        # k(x_i,y_j) only, then multiply the result with V.
        if operation == "*" and other.formula[:3] == "Var" and other.ndim > 100:
            res.rec_multVar_highdim = (self, other)

        return res

    def ternary(
        self, other1, other2, operation, dimres=None, dimcheck="sameor1", opt_arg=None
    ):
        r"""Symbolically applies **operation** to **self**, with optional arguments if needed.

        Keyword args:
          - dimres (int): May be used to specify the dimension of the output **result**.
          - is_operator (bool, default=False): May be used to specify if **operation** is
            an operator like ``+``, ``-`` or a "genuine" function.
          - dimcheck (string): shall we check the input dimensions?
            Supported values are ``"same"``, ``"sameor1"``, or **None**.
        """
        # If needed, convert float numbers / lists / arrays / tensors to LazyTensors:
        if not hasattr(other1, "__GenericLazyTensor__"):
            other1 = self.lt_constructor(other1)

        if not hasattr(other2, "__GenericLazyTensor__"):
            other2 = self.lt_constructor(other2)

        # we must prevent any operation if self, other1 or other2 is the output of a reduction operation,
        # i.e. if it has a reduction_op field
        if (
            hasattr(self, "reduction_op")
            or hasattr(other1, "reduction_op")
            or hasattr(other2, "reduction_op")
        ):
            raise ValueError(
                "One of the inputs is a 'reduced' LazyTensor, no operation can be applied to it. "
            )

        # By default, the dimension of the output variable is the max of the three operands:
        if not dimres:
            dimres = max(self.ndim, other1.ndim, other2.ndim)

        if dimcheck == "same":
            if (self.ndim != other1.ndim) or (self.ndim != other2.ndim):
                raise ValueError(
                    "Operation {} expects inputs of the same dimension. ".format(
                        operation
                    )
                    + "Received {}, {} and {}.".format(
                        self.ndim, other1.ndim, other2.ndim
                    )
                )

        elif dimcheck == "sameor1":
            if not same_or_one_test(self.ndim, other1.ndim, other2.ndim):
                raise ValueError(
                    "Operation {} expects inputs of the same dimension or dimension 1. ".format(
                        operation
                    )
                    + "Received {}, {} and {}.".format(
                        self.ndim, other1.ndim, other2.ndim
                    )
                )

        elif dimcheck != None:
            raise ValueError("incorrect dimcheck keyword in binary operation")

        res = self.join(
            other1.join(other2)
        )  # Merge the attributes and variables of operands
        res.ndim = dimres

        if opt_arg is not None:
            if hasattr(opt_arg, "__GenericLazyTensor__"):
                opt_arg = opt_arg.formula
            res.formula = "{}({}, {}, {}, {})".format(
                operation, self.formula, other1.formula, other2.formula, opt_arg
            )
        else:
            res.formula = "{}({}, {}, {})".format(
                operation, self.formula, other1.formula, other2.formula
            )

        return res

    # Prototypes for reduction operations  =====================================

    def reduction(
        self,
        reduction_op,
        other=None,
        opt_arg=None,
        axis=None,
        dim=None,
        call=True,
        is_complex=None,
        **kwargs
    ):
        r"""
        Applies a reduction to a :class:`LazyTensor`. This method is used internally by the LazyTensor class.
        Args:
            reduction_op (string): the string identifier of the reduction, which will be passed to the KeOps routines.

        Keyword Args:
          other: May be used to specify some **weights** ; depends on the reduction.
          opt_arg: typically, some integer needed by ArgKMin reductions ; depends on the reduction.
          axis (integer): The axis with respect to which the reduction should be performed.
            Supported values are **nbatchdims** and **nbatchdims + 1**, where **nbatchdims** is the number of "batch" dimensions before the last three
            (:math:`i` indices, :math:`j` indices, variables' dimensions).
          dim (integer): alternative keyword for the **axis** argument.
          call (True or False): Should we actually perform the reduction on the current variables?
            If **True**, the returned object will be a NumPy array or a PyTorch tensor.
            Otherwise, we simply return a callable :class:`LazyTensor` that may be used
            as a :mod:`pykeops.numpy.Genred` or :mod:`pykeops.torch.Genred` function
            on arbitrary tensor data.
          backend (string): Specifies the map-reduce scheme,
            as detailed in the documentation of the :mod:`Genred <pykeops.torch.Genred>` module.
          device_id (int, default=-1): Specifies the GPU that should be used
            to perform the computation; a negative value lets your system
            choose the default GPU. This parameter is only useful if your
            system has access to several GPUs.
          ranges (6-uple of IntTensors, None by default):
            Ranges of integers that specify a
            :doc:`block-sparse reduction scheme <../../sparsity>`
            as detailed in the documentation of the :mod:`Genred <pykeops.torch.Genred>` module.
            If **None** (default), we simply use a **dense Kernel matrix**
            as we loop over all indices
            :math:`i\in[0,M)` and :math:`j\in[0,N)`.
          dtype_acc (string, default ``"auto"``): type for accumulator of reduction, before casting to dtype.
            It improves the accuracy of results in case of large sized data, but is slower.
            Default value "auto" will set this option to the value of dtype. The supported values are:
              - **dtype_acc** = ``"float16"`` : allowed only if dtype is "float16".
              - **dtype_acc** = ``"float32"`` : allowed only if dtype is "float16" or "float32".
              - **dtype_acc** = ``"float64"`` : allowed only if dtype is "float32" or "float64"..
          use_double_acc (bool, default False): same as setting dtype_acc="float64" (only one of the two options can be set)
            If True, accumulate results of reduction in float64 variables, before casting to float32.
            This can only be set to True when data is in float32 or float64.
            It improves the accuracy of results in case of large sized data, but is slower.
          sum_scheme (string, default ``"auto"``): method used to sum up results for reductions. This option may be changed only
            when reduction_op is one of: "Sum", "MaxSumShiftExp", "LogSumExp", "Max_SumShiftExpWeight", "LogSumExpWeight", "SumSoftMaxWeight".
            Default value "auto" will set this option to "block_red" for these reductions. Possible values are:
              - **sum_scheme** =  ``"direct_sum"``: direct summation
              - **sum_scheme** =  ``"block_sum"``: use an intermediate accumulator in each block before accumulating
                in the output. This improves accuracy for large sized data.
              - **sum_scheme** =  ``"kahan_scheme"``: use Kahan summation algorithm to compensate for round-off errors. This improves
                accuracy for large sized data.
          enable_chunks (bool, default True): enable automatic selection of special "chunked" computation mode for accelerating reductions
                                with formulas involving large dimension variables.
          out (2d NumPy array or PyTorch Tensor, None by default): The output numerical array, for in-place computation.
              If provided, the output array should all have the same ``dtype``, be **contiguous** and be stored on
              the **same device** as the arguments. Moreover it should have the correct shape for the output.
        """

        if is_complex is None:
            if other is None:
                is_complex = self.is_complex
            else:
                is_complex = self.is_complex or other.is_complex

        if axis is None:
            axis = dim  # NumPy uses axis, PyTorch uses dim...
        if axis - self.nbatchdims not in (0, 1):
            raise ValueError(
                "Reductions must be called with 'axis' (or 'dim') equal to the number of batch dimensions + 0 or 1."
            )

        if other is None:
            res = self.init(is_complex=is_complex)  # ~ self.copy()
            res.formula2 = None
        else:
            res = self.join(other, is_complex=is_complex)
            res.formula2 = other.formula

        res.formula = self.formula
        res.reduction_op = reduction_op
        res.axis = axis - self.nbatchdims
        res.opt_arg = opt_arg

        kwargs_init, kwargs_call = self.separate_kwargs(kwargs)

        res.kwargs = kwargs_call
        res.ndim = self.ndim
        if reduction_op == "Sum" and hasattr(self, "rec_multVar_highdim"):
            if res.axis != self.rec_multVar_highdim[1].axis:
                return (
                    self.rec_multVar_highdim[0].sum(axis=axis)
                    * self.rec_multVar_highdim[1].variables[0]
                )
            res.rec_multVar_highdim = id(self.rec_multVar_highdim[1].variables[0])
        else:
            res.rec_multVar_highdim = None
        if res._dtype is not None:
            res.fixvariables()  # Turn the "id(x)" numbers into consecutive labels
            # "res" now becomes a callable object:
            res.callfun = res.Genred(
                res.formula,
                [],
                reduction_op=res.reduction_op,
                axis=res.axis,
                opt_arg=res.opt_arg,
                formula2=res.formula2,
                **kwargs_init,
                rec_multVar_highdim=res.rec_multVar_highdim
            )
        if call and len(res.symbolic_variables) == 0 and res._dtype is not None:
            return res()
        else:
            return res

    def solve(self, other, var=None, call=True, **kwargs):
        r"""
        Solves a positive definite linear system of the form ``sum(self) = other`` or ``sum(self*var) = other`` , using a conjugate gradient solver.

        Args:
          self (:class:`LazyTensor`): KeOps variable that encodes a symmetric positive definite matrix / linear operator.
          other (:class:`LazyTensor`): KeOps variable that encodes the second member of the equation.

        Keyword args:
          var (:class:`LazyTensor`):
            If **var** is **None**, **solve** will return the solution
            of the ``self * var = other`` equation.
            Otherwise, if **var** is a KeOps symbolic variable, **solve** will
            assume that **self** defines an expression that is linear
            with respect to **var** and solve the equation ``self(var) = other``
            with respect to **var**.
          alpha (float, default=1e-10): Non-negative **ridge regularization** parameter.
          call (bool): If **True** and if no other symbolic variable than
            **var** is contained in **self**, **solve** will return a tensor
            solution of our linear system. Otherwise **solve** will return
            a callable :class:`LazyTensor`.
          backend (string): Specifies the map-reduce scheme,
            as detailed in the documentation of the :mod:`Genred <pykeops.torch.Genred>` module.
          device_id (int, default=-1): Specifies the GPU that should be used
            to perform the computation; a negative value lets your system
            choose the default GPU. This parameter is only useful if your
            system has access to several GPUs.
          ranges (6-uple of IntTensors, None by default):
            Ranges of integers that specify a
            :doc:`block-sparse reduction scheme <../../sparsity>`
            as detailed in the documentation of the :mod:`Genred <pykeops.torch.Genred>` module.
            If **None** (default), we simply use a **dense Kernel matrix**
            as we loop over all indices :math:`i\in[0,M)` and :math:`j\in[0,N)`.
          dtype_acc (string, default ``"auto"``): type for accumulator of reduction, before casting to dtype.
            It improves the accuracy of results in case of large sized data, but is slower.
            Default value "auto" will set this option to the value of dtype. The supported values are:
              - **dtype_acc** = ``"float16"`` : allowed only if dtype is "float16".
              - **dtype_acc** = ``"float32"`` : allowed only if dtype is "float16" or "float32".
              - **dtype_acc** = ``"float64"`` : allowed only if dtype is "float32" or "float64"..
          use_double_acc (bool, default False): same as setting dtype_acc="float64" (only one of the two options can be set)
            If True, accumulate results of reduction in float64 variables, before casting to float32.
            This can only be set to True when data is in float32 or float64.
            It improves the accuracy of results in case of large sized data, but is slower.
          sum_scheme (string, default ``"auto"``): method used to sum up results for reductions. This option may be changed only
            when reduction_op is one of: "Sum", "MaxSumShiftExp", "LogSumExp", "Max_SumShiftExpWeight", "LogSumExpWeight", "SumSoftMaxWeight".
            Default value "auto" will set this option to "block_red" for these reductions. Possible values are:
              - **sum_scheme** =  ``"direct_sum"``: direct summation
              - **sum_scheme** =  ``"block_sum"``: use an intermediate accumulator in each block before accumulating
                in the output. This improves accuracy for large sized data.
              - **sum_scheme** =  ``"kahan_scheme"``: use Kahan summation algorithm to compensate for round-off errors. This improves
                accuracy for large sized data.
            enable_chunks (bool, default True): enable automatic selection of special "chunked" computation mode for accelerating reductions
                                with formulas involving large dimension variables.

        .. warning::

            Please note that **no check** of symmetry and definiteness will be
            performed prior to our conjugate gradient descent.
        """

        if not hasattr(other, "__GenericLazyTensor__"):
            other = self.lt_constructor(
                x=other, axis=0
            )  # a vector is normally indexed by "i"

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
            var = self.lt_constructor((varindex, other.ndim, axis))
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

        kwargs_init, res.kwargs = self.separate_kwargs(kwargs)

        res.ndim = self.ndim

        if other.ndim > 100:
            res.rec_multVar_highdim = varindex
        else:
            res.rec_multVar_highdim = None

        if res._dtype is not None:
            res.fixvariables()
            res.callfun = res.KernelSolve(
                res.formula,
                [],
                res.varformula,
                res.axis,
                **kwargs_init,
                rec_multVar_highdim=res.rec_multVar_highdim
            )

        # we call if call=True, if other is not symbolic, and if the dtype is set
        if call and len(other.symbolic_variables) == 0 and res._dtype is not None:
            return res()
        else:
            return res

    def __call__(self, *args, **kwargs):
        r"""
        Executes a :mod:`Genred <pykeops.torch.Genred>` or :mod:`KernelSolve <pykeops.torch.KernelSolve>` call on the input data, as specified by **self.formula** .
        """
        if not hasattr(self, "reduction_op"):
            raise ValueError(
                "A LazyTensor object may be called only if it corresponds to the output of a reduction operation or solve operation."
            )

        self.kwargs.update(kwargs)

        if self.ranges is not None and "ranges" not in self.kwargs:
            self.kwargs.update({"ranges": self.ranges})

        if self.backend is not None and "backend" not in self.kwargs:
            self.kwargs.update({"backend": self.backend})

        if (
            self._dtype is None
        ):  # This can only happen if we haven't encountered 2D or 3D arrays just yet...
            self.get_tools()

            self._dtype = self.tools.dtypename(
                self.tools.dtype(args[0])
            )  # crash if LazyTensor is called
            self.fixvariables()

            kwargs_init, self.kwargs = self.separate_kwargs(self.kwargs)

            if self.reduction_op == "Solve":
                self.callfun = self.KernelSolve(
                    self.formula,
                    [],
                    self.formula2,
                    self.axis,
                    self._dtype,
                    **kwargs_init,
                    rec_multVar_highdim=self.rec_multVar_highdim
                )
            else:
                self.callfun = self.Genred(
                    self.formula,
                    [],
                    self.reduction_op,
                    self.axis,
                    self._dtype,
                    self.opt_arg,
                    self.formula2,
                    **kwargs_init,
                    rec_multVar_highdim=self.rec_multVar_highdim
                )

        if self.reduction_op == "Solve" and len(self.other.symbolic_variables) == 0:
            # here args should be empty, according to our rule
            if args != ():
                raise ValueError("no input required")
            # we replace by other
            args = (self.other.variables[0],)

        return self.callfun(*args, *self.variables, **self.kwargs)

    def __str__(self):
        r"""
        Returns a verbose string identifier.
        """
        tmp = self.init(is_complex=self.is_complex)  # ~ self.copy()
        tmp.formula = self.formula
        tmp.formula2 = None if not hasattr(self, "formula2") else self.formula2

        tmp.fixvariables()  # Replace Var(id(x),...) with consecutive labels

        string = "KeOps LazyTensor\n    formula: {}".format(tmp.formula)
        if len(self.symbolic_variables) > 0:
            string += "\n    symbolic variables: Var{}".format(
                self.symbolic_variables[0]
            )
            for var in self.symbolic_variables[1:]:
                string += ", Var{}".format(var)

        string += "\n    shape: {}".format(self.shape)

        if hasattr(self, "reduction_op"):
            string += "\n    reduction: {} (axis={})".format(
                self.reduction_op, self.axis
            )
            if tmp.formula2 is not None:
                string += "\n        formula2: {}".format(tmp.formula2)
            if hasattr(self, "opt_arg") and self.opt_arg is not None:
                string += "\n        opt_arg: {}".format(self.opt_arg)
        return string

    @property
    def dtype(self):
        return self._dtype

    @property
    def _shape(self):
        r"""Returns the internal shape of the LazyTensor."""
        btch = () if self.batchdims is None else self.batchdims
        ni = 1 if self.ni is None else self.ni
        nj = 1 if self.nj is None else self.nj
        ndim = 1 if self.ndim is None else self.ndim
        return btch + (ni, nj, ndim)

    @property
    def shape(self):
        r"""Returns the shape of the LazyTensor"""
        s = self._shape
        if s[-1] == 1:
            return s[:-1]
        else:
            return s

    def dim(self):
        r"""
        Just as in PyTorch, returns the number of dimensions of a :class:`LazyTensor`.
        """
        return len(self._shape)

    @property
    def nbatchdims(self):
        return 0 if self.batchdims is None else len(self.batchdims)

    # List of supported operations  ============================================

    # N.B.: This flag prevents NumPy (and also PyTorch ?) from overriding
    #       the KeOps implementations of __radd__, __rdiv___, etc. written below.
    #       For instance, if x is a NumPy array and y is a KeOps LazyTensor,
    #       writing  "x+y"  will call y.__radd__(x) (LazyTensor method) instead
    #       of x.__add__(y) (NumPy method)
    __array_ufunc__ = None

    # Arithmetics --------------------------------------------------------------

    def addop(self, other, **kwargs):
        return self.binary(other, "+", is_operator=True, **kwargs)

    def __add__(self, other):
        r"""
        Broadcasted addition operator - a binary operation.

        ``x + y`` returns a :class:`LazyTensor` that encodes,
        symbolically, the addition of ``x`` and ``y``.
        """
        if is_scalar_and_equals(other, 0):
            return self
        elif is_complex_lazytensor(other) and not is_complex_lazytensor(self):
            return self.real2complex().addop(other)
        else:
            return self.addop(other)

    def __radd__(self, other):
        r"""
        Broadcasted addition operator - a binary operation.

        ``x + y`` returns a :class:`LazyTensor` that encodes,
        symbolically, the addition of ``x`` and ``y``.
        """
        if is_scalar_and_equals(other, 0):
            return self
        else:
            return self.addop(other, rversion=True)

    def subop(self, other, **kwargs):
        return self.binary(other, "-", is_operator=True, **kwargs)

    def __sub__(self, other):
        r"""
        Broadcasted subtraction operator - a binary operation.

        ``x - y`` returns a :class:`LazyTensor` that encodes,
        symbolically, the subtraction of ``x`` and ``y``.
        """
        if is_scalar_and_equals(other, 0):
            return self
        elif is_complex_lazytensor(other) and not is_complex_lazytensor(self):
            return self.real2complex().subop(other)
        else:
            return self.subop(other)

    def __rsub__(self, other):
        r"""
        Broadcasted subtraction operator - a binary operation.

        ``x - y`` returns a :class:`LazyTensor` that encodes,
        symbolically, the subtraction of ``x`` and ``y``.
        """
        if is_scalar_and_equals(other, 0):
            return self.unary("Minus")
        else:
            return self.subop(other, rversion=True)

    def mulop(self, other, **kwargs):
        return self.binary(other, "*", is_operator=True, **kwargs)

    def __mul__(self, other):
        r"""
        Broadcasted element-wise product - a binary operation.

        ``x * y`` returns a :class:`LazyTensor` that encodes, symbolically,
        the elementwise product of ``x`` and ``y``.
        """
        if is_scalar_and_equals(other, 0):
            return 0
        elif is_scalar_and_equals(other, 1):
            return self
        elif is_scalar_and_equals(other, -1):
            return self.unary("Minus")
        elif is_complex_lazytensor(other) and not is_complex_lazytensor(self):
            return other.mulop(self)
        elif self.tools.detect_complex(other) and not is_complex_lazytensor(self):
            return self.lt_constructor(other).mulop(self)
        else:
            return self.mulop(other)

    def __rmul__(self, other):
        r"""
        Broadcasted element-wise product - a binary operation.

        ``x * y`` returns a :class:`LazyTensor` that encodes, symbolically,
        the elementwise product of ``x`` and ``y``.
        """
        if is_scalar_and_equals(other, 0):
            return 0
        elif is_scalar_and_equals(other, 1):
            return self
        elif is_scalar_and_equals(other, -1):
            return self.unary("Minus")
        elif self.tools.detect_complex(other) and not is_complex_lazytensor(self):
            return self.real2complex().mulop(self.lt_constructor(other))
        else:
            return self.mulop(other, rversion=True)

    def divop(self, other, **kwargs):
        return self.binary(other, "/", is_operator=True, **kwargs)

    def __truediv__(self, other):
        r"""
        Broadcasted element-wise division - a binary operation.

        ``x / y`` returns a :class:`LazyTensor` that encodes, symbolically,
        the elementwise division of ``x`` by ``y``.
        """
        if is_scalar_and_equals(other, 1):
            return self
        elif is_complex_lazytensor(other) and not is_complex_lazytensor(self):
            return self.real2complex().divop(other)
        else:
            return self.divop(other)

    def __rtruediv__(self, other):
        r"""
        Broadcasted element-wise division - a binary operation.

        ``x / y`` returns a :class:`LazyTensor` that encodes, symbolically,
        the elementwise division of ``x`` by ``y``.
        """
        if is_scalar_and_equals(other, 0):
            return 0
        elif is_scalar_and_equals(other, 1):
            return self.unary("Inv")
        else:
            return self.divop(other, rversion=True)

    def __or__(self, other):
        r"""
        Euclidean scalar product - a binary operation.

        ``(x|y)`` returns a :class:`LazyTensor` that encodes, symbolically,
        the scalar product of ``x`` and ``y`` which are assumed to have the same shape.
        """
        return self.binary(other, "|", is_operator=True, dimres=1, dimcheck="same")

    def __ror__(self, other):
        r"""
        Euclidean scalar product - a binary operation.

        ``(x|y)`` returns a :class:`LazyTensor` that encodes, symbolically,
        the scalar product of ``x`` and ``y`` which are assumed to have the same shape.
        """
        return self.binary(
            other, "|", is_operator=True, dimres=1, dimcheck="same", rversion=True
        )

    # Unary arithmetics --------------------------------------------------------

    def __abs__(self):
        r"""
        Element-wise absolute value - a unary operation.

        ``abs(x)`` returns a :class:`LazyTensor` that encodes, symbolically,
        the element-wise absolute value of ``x``.
        """
        return self.unary("Abs")

    def abs(self):
        r"""
        Element-wise absolute value - a unary operation.

        ``x.abs()`` returns a :class:`LazyTensor` that encodes, symbolically,
        the element-wise absolute value of ``x``.
        """
        return abs(self)

    def __neg__(self):
        r"""
        Element-wise minus - a unary operation.

        ``-x`` returns a :class:`LazyTensor` that encodes, symbolically,
        the element-wise opposite of ``x``.
        """
        return self.unary("Minus")

    # Simple functions ---------------------------------------------------------

    def exp(self):
        r"""
        Element-wise exponential - a unary operation.

        ``x.exp()`` returns a :class:`LazyTensor` that encodes, symbolically,
        the element-wise exponential of ``x``.
        """
        return self.unary("Exp")

    def log(self):
        r"""
        Element-wise logarithm - a unary operation.

        ``x.log()`` returns a :class:`LazyTensor` that encodes, symbolically,
        the element-wise logarithm of ``x``.
        """
        return self.unary("Log")

    def xlogx(self):
        r"""
        Element-wise x*log(x) function - a unary operation.

        ``x.xlogx()`` returns a :class:`LazyTensor` that encodes, symbolically,
        the element-wise ``x`` times logarithm of ``x`` (with value 0 at 0).
        """
        return self.unary("XLogX")

    def cos(self):
        r"""
        Element-wise cosine - a unary operation.

        ``x.cos()`` returns a :class:`LazyTensor` that encodes, symbolically,
        the element-wise cosine of ``x``.
        """
        return self.unary("Cos")

    def sin(self):
        r"""
        Element-wise sine - a unary operation.

        ``x.sin()`` returns a :class:`LazyTensor` that encodes, symbolically,
        the element-wise sine of ``x``.
        """
        return self.unary("Sin")

    def sinxdivx(self):
        r"""
        Element-wise sin(x)/x function - a unary operation.

        ``x.sinxdivx()`` returns a :class:`LazyTensor` that encodes, symbolically,
        the element-wise sinxdivx function  of ``x``.
        """
        return self.unary("SinXDivX")

    def sinc(self):
        r"""
        Element-wise sinc(x) = sin(pi x) / (pi x) function - a unary operation.

        ``x.sinc()`` returns a :class:`LazyTensor` that encodes, symbolically,
        the element-wise sinc function  of ``x``.
        """
        return (math.pi * self).sinxdivx()

    def asin(self):
        r"""
        Element-wise arcsine - a unary operation.

        ``x.asin()`` returns a :class:`LazyTensor` that encodes, symbolically,
        the element-wise arcsine of ``x``.
        """
        return self.unary("Asin")

    def acos(self):
        r"""
        Element-wise arccosine - a unary operation.

        ``x.acos()`` returns a :class:`LazyTensor` that encodes, symbolically,
        the element-wise arccosine of ``x``.
        """
        return self.unary("Acos")

    def atan(self):
        r"""
        Element-wise arctangent - a unary operation.

        ``x.atan()`` returns a :class:`LazyTensor` that encodes, symbolically,
        the element-wise arctangent of ``x``.
        """
        return self.unary("Atan")

    def atan2(self, other):
        r"""
        Element-wise atan2 - a binary operation.

        ``y.atan2(x)`` returns a :class:`LazyTensor` that encodes, symbolically,
        the element-wise atan2 of ``x`` and ``y``.
        """
        return self.binary(other, "Atan2", dimcheck="sameor1")

    def sqrt(self):
        r"""
        Element-wise square root - a unary operation.

        ``x.sqrt()`` returns a :class:`LazyTensor` that encodes, symbolically,
        the element-wise square root of ``x``.
        """
        return self.unary("Sqrt")

    def rsqrt(self):
        r"""
        Element-wise inverse square root - a unary operation.

        ``x.rsqrt()`` returns a :class:`LazyTensor` that encodes, symbolically,
        the element-wise inverse square root of ``x``.
        """
        return self.unary("Rsqrt")

    def __pow__(self, other):
        r"""
        Broadcasted element-wise power operator - a binary operation.

        ``x**y`` returns a :class:`LazyTensor` that encodes, symbolically,
        the element-wise value of ``x`` to the power ``y``.

        Note:
          - if **y = 2**, ``x**y`` relies on the ``"Square"`` KeOps operation;
          - if **y = 0.5**, ``x**y`` uses on the ``"Sqrt"`` KeOps operation;
          - if **y = -0.5**, ``x**y`` uses on the ``"Rsqrt"`` KeOps operation.
        """
        if type(other) == int:
            return (
                self.unary("Square") if other == 2 else self.unary("Pow", opt_arg=other)
            )

        elif type(other) == float:
            if other == 0.5:
                return self.unary("Sqrt")
            elif other == -0.5:
                return self.unary("Rsqrt")
            else:
                other = self.lt_constructor(other)

        if hasattr(other, "__GenericLazyTensor__"):
            if other.ndim == 1 or other.ndim == self.ndim:
                return self.binary(other, "Powf", dimcheck=None)
            else:
                raise ValueError(
                    "Incompatible dimensions for the LazyTensor and its exponent: "
                    + "{} and {}.".format(self.ndim, other.ndim)
                )
        else:
            raise ValueError(
                "The exponent should be an integer, a float number or a LazyTensor."
            )

    def power(self, other):
        r"""
        Broadcasted element-wise power operator - a binary operation.

        ``pow(x,y)`` is equivalent to ``x**y``.
        """
        return self**other

    def square(self):
        r"""
        Element-wise square - a unary operation.

        ``x.square()`` is equivalent to ``x**2`` and returns a :class:`LazyTensor`
        that encodes, symbolically, the element-wise square of ``x``.
        """
        return self.unary("Square")

    def sign(self):
        r"""
        Element-wise sign in {-1,0,+1} - a unary operation.

        ``x.sign()`` returns a :class:`LazyTensor` that encodes, symbolically,
        the element-wise sign of ``x``.
        """
        return self.unary("Sign")

    def step(self):
        r"""
        Element-wise step function - a unary operation.

        ``x.step()`` returns a :class:`LazyTensor` that encodes, symbolically,
        the element-wise sign of ``x``.
        """
        return self.unary("Step")

    def relu(self):
        r"""
        Element-wise ReLU function - a unary operation.

        ``x.relu()`` returns a :class:`LazyTensor` that encodes, symbolically,
        the element-wise positive part of ``x``.
        """
        return self.unary("ReLU")

    def clamp(self, other1, other2):
        r"""
        Element-wise Clamp function - a ternary operation.

        ``x.clamp(a,b)`` returns a :class:`LazyTensor` that encodes, symbolically,
        the element-wise clamping of ``x`` in ``(a,b)``. Broadcasting rules apply.
        a and b may be fixed integers or floats, or other LazyTensors.
        """
        if (type(other1) == int) and (type(other2) == int):
            return self.unary("ClampInt", opt_arg=other1, opt_arg2=other2)
        else:
            return self.ternary(other1, other2, "Clamp", dimcheck="sameor1")

    def ifelse(self, other1, other2):
        r"""
        Element-wise if-else function - a ternary operation.

        ``x.ifelse(a,b)`` returns a :class:`LazyTensor` that encodes, symbolically,
        ``a`` where ``x >= 0`` and ``b`` where ``x < 0``.  Broadcasting rules apply.
        a and b may be fixed integers or floats, or other LazyTensors.
        """
        return self.ternary(other1, other2, "IfElse", dimcheck="sameor1")

    def mod(self, modulus, offset=0):
        r"""
        Element-wise modulo with offset function - a ternary operation.

        ``x.mod(a,b)`` returns a :class:`LazyTensor` that encodes, symbolically,
        the element-wise modulo of ``x`` with modulus ``a`` and offset ``b``.
        By default b=0, so that x.mod(a) becomes equivalent to the NumPy function mod.
        Broadcasting rules apply. a and b are fixed integers or float.
        """
        return self.ternary(modulus, offset, "Mod", dimcheck="sameor1")

    def round(self, other=0):
        r"""
        Element-wise rounding function - a unary operation.

        ``x.round(d)`` returns a :class:`LazyTensor` that encodes, symbolically,
        the element-wise rounding of ``x`` to d decimal places. d is int.
        """
        return self.unary("Round", opt_arg=other)

    def sqnorm2(self):
        r"""
        Squared Euclidean norm - a unary operation.

        ``x.sqnorm2()`` returns a :class:`LazyTensor` that encodes, symbolically,
        the squared Euclidean norm of a vector ``x``.
        """
        return self.unary("SqNorm2", dimres=1)

    def norm2(self):
        r"""
        Euclidean norm - a unary operation.

        ``x.norm2()`` returns a :class:`LazyTensor` that encodes, symbolically,
        the Euclidean norm of a vector ``x``.
        """
        return self.unary("Norm2", dimres=1)

    def norm(self, dim):
        r"""
        Euclidean norm - a unary operation.

        ``x.norm(-1)`` is equivalent to ``x.norm2()`` and returns a
        :class:`LazyTensor` that encodes, symbolically, the Euclidean norm of a vector ``x``.
        """
        if dim not in [-1, len(self._shape) - 1]:
            raise ValueError("KeOps only supports norms over the last dimension.")
        return self.norm2()

    def normalize(self):
        r"""
        Vector normalization - a unary operation.

        ``x.normalize()`` returns a :class:`LazyTensor` that encodes, symbolically,
        a vector ``x`` divided by its Euclidean norm.
        """
        return self.unary("Normalize")

    def sqdist(self, other):
        r"""
        Squared distance - a binary operation.

        ``x.sqdist(y)`` returns a :class:`LazyTensor` that encodes, symbolically,
        the squared Euclidean distance between two vectors ``x`` and ``y``.
        """
        return self.binary(other, "SqDist", dimres=1)

    def weightedsqnorm(self, other):
        r"""
        Weighted squared norm of a LazyTensor ``x`` - a binary operation.

        ``x.weightedsqnorm(s)`` returns a :class:`LazyTensor` that encodes, symbolically,
        the weighted squared Norm of a vector ``x`` with weights stored in the LazyTensor ``s``- see
        the :doc:`main reference page <../../../api/math-operations>` for details.
        """
        if not hasattr(other, "__GenericLazyTensor__"):
            other = self.lt_constructor(other)

        if other.ndim not in (1, self.ndim, self.ndim**2):
            raise ValueError(
                "Squared norm weights should be of size 1 (scalar), "
                + "D (diagonal) or D^2 (full symmetric tensor), but received "
                + "{} with D={}.".format(other.ndim, self.ndim)
            )

        return self.binary(
            other, "WeightedSqNorm", dimres=1, dimcheck=None, rversion=True
        )

    def weightedsqdist(self, g, s):
        r"""
        Weighted squared distance.

        ``x.weightedsqdist(y, s)`` is equivalent to ``(x - y).weightedsqnorm(s)``.
        """
        if not hasattr(g, "__GenericLazyTensor__"):
            g = self.lt_constructor(g)
        if not hasattr(s, "__GenericLazyTensor__"):
            s = self.lt_constructor(s)

        return (self - g).weightedsqnorm(s)

    def elem(self, i):
        r"""
        Indexing of a vector - a unary operation.

        ``x.elem(i)`` returns a :class:`LazyTensor` that encodes, symbolically,
        the i-th element ``x[i]`` of the vector ``x``.
        """
        if type(i) is not int:
            raise ValueError("Elem indexing is only supported for integer indices.")
        if i < 0 or i >= self.ndim:
            raise ValueError(
                "Index i={} is out of bounds [0,D) = [0,{}).".format(i, self.ndim)
            )
        return self.unary("Elem", dimres=1, opt_arg=i)

    def extract(self, i, d):
        r"""
        Range indexing - a unary operation.

        ``x.extract(i, d)`` returns a :class:`LazyTensor` that encodes, symbolically,
        the sub-vector ``x[i:i+d]`` of the vector ``x``.
        """
        if (type(i) is not int) or (type(d) is not int):
            raise ValueError("Indexing is only supported for integer indices.")
        if i < 0 or i >= self.ndim:
            raise ValueError("Starting index is out of bounds.")
        if d < 1 or i + d > self.ndim:
            raise ValueError("Slice dimension is out of bounds.")
        return self.unary("Extract", dimres=d, opt_arg=i, opt_arg2=d)

    def __getitem__(self, key):
        r"""
        Element or range indexing - a unary operation.

        ``x[key]`` redirects to the :meth:`elem` or :meth:`extract` methods, depending on the ``key`` argument.
        Supported values are:

            - an integer ``k``, in which case ``x[key]``
              redirects to ``elem(x,k)``,
            - a tuple ``..,:,:,k`` with ``k`` an integer,
              which is equivalent to the case above,
            - a slice of the form ``k:l``, ``k:`` or ``:l``, with ``k``
              and ``l`` two integers, in which case ``x[key]`` redirects to ``extract(x,k,l-k)``,
            - a tuple of slices of the form ``..,:,:,k:l``, ``..,:,:,k:`` or ``..,:,:,:l``,
              with ``k`` and ``l`` two integers, which are equivalent to the case above.
        """
        # we allow only these forms:
        #    [..,:,:,k], [..,:,:,k:l], [..,:,:,k:], [..,:,:,:l]
        #    or equivalent [k], [k:l], [k:], [:l]
        if isinstance(key, tuple):
            if len(key) == len(self._shape) and key[:-1] == (slice(None),) * (
                len(self._shape) - 1
            ):
                key = key[-1]
            else:
                raise ValueError(
                    "LazyTensors only support indexing with respect to their last dimension."
                )

        if isinstance(key, slice):
            if not key.step in [None, 1]:
                raise ValueError(
                    "LazyTensors do not support sliced indexing with stepsizes > 1."
                )
            if key.start is None:
                key = slice(0, key.stop)
            if key.stop is None:
                key = slice(key.start, self.ndim)
            return self.extract(key.start, key.stop - key.start)
        elif isinstance(key, int):
            return self.elem(key)
        else:
            raise ValueError(
                "LazyTensors only support indexing with integers and vanilla python slices."
            )

    def one_hot(self, D):
        r"""
        Encodes a (rounded) scalar value as a one-hot vector of dimension D.

        ``x.one_hot(D)`` returns a :class:`LazyTensor` that encodes, symbolically,
        a vector of length D whose round(x)-th coordinate is equal to 1, and the other ones to zero.
        """
        if type(D) is not int:
            raise ValueError(
                "One-hot encoding expects an integer dimension of the output vector."
            )
        if self.ndim != 1:
            raise ValueError("One-hot encoding is only supported for scalar formulas.")

        return self.unary("OneHot", dimres=D, opt_arg=D)

    def concat(self, other):
        r"""
        Concatenation of two :class:`LazyTensor` - a binary operation.

        ``x.concat(y)`` returns a :class:`LazyTensor` that encodes, symbolically,
        the concatenation of ``x`` and ``y`` along their last dimension.
        """
        return self.binary(
            other, "Concat", dimres=(self.ndim + other.ndim), dimcheck=None
        )

    @staticmethod
    def concatenate(tuple_of_lt, axis=-1):
        r"""
        Concatenation of a tuple of :class:`GenericLazyTensor`.

        ``GenericLazyTensor.concatenate( (x_1, x_2, ..., x_n), -1)`` returns a :class:`GenericLazyTensor` that encodes, symbolically,
        the concatenation of ``x_1``, ``x_2``, ..., ``x_n`` along their last dimension.
        Note that **axis** should be equal to -1 or 2 (if the ``x_i``'s are 3D GenericLazyTensor):
        GenericLazyTensors only support concatenation and indexing operations with respect
        to the last dimension.
        """
        if isinstance(tuple_of_lt, tuple):
            if len(tuple_of_lt) == 0:
                raise ValueError("Received an empty tuple of LazyTensors.")
            elif hasattr(tuple_of_lt[0], "__GenericLazyTensor__"):
                if axis not in [-1, len(tuple_of_lt[0]._shape) - 1]:
                    raise ValueError(
                        "LazyTensor only supports concatenation along the last axis."
                    )
                if len(tuple_of_lt) == 1:
                    return tuple_of_lt[0]
                elif len(tuple_of_lt) == 2:
                    return tuple_of_lt[0].concat(tuple_of_lt[1])
                else:
                    return GenericLazyTensor.concatenate(
                        (tuple_of_lt[0].concat(tuple_of_lt[1]),) + tuple_of_lt[2:],
                        axis=-1,
                    )
        else:
            raise ValueError(
                "LazyTensor.concatenate is implemented on *tuples* of LazyTensors."
            )

    @staticmethod
    def cat(tuple_of_lt, dim):
        r"""
        Concatenation of a tuple of LazyTensors.

        ``LazyTensor.cat( (x_1, x_2, ..., x_n), -1)``
        is a PyTorch-friendly alias for ``LazyTensor.concatenate( (x_1, x_2, ..., x_n), -1)``;
        just like indexing operations, it is only supported along the last dimension.
        """
        return GenericLazyTensor.concatenate(tuple_of_lt, dim)

    def matvecmult(self, other):
        r"""
        Matrix-vector product - a binary operation.

        If ``x._shape[-1] == A*B`` and ``y._shape[-1] == B``,
        ``z = x.matvecmult(y)`` returns a :class:`GenericLazyTensor`
        such that ``z._shape[-1] == A`` which encodes, symbolically,
        the matrix-vector product of ``x`` and ``y`` along their last dimension.
        For details, please check the documentation of the KeOps operation ``"MatVecMult"`` in
        the :doc:`main reference page <../../../api/math-operations>`.
        """
        return self.binary(
            other, "MatVecMult", dimres=(self.ndim // other.ndim), dimcheck=None
        )

    def vecmatmult(self, other):
        r"""
        Vector-matrix product - a binary operation.

        If ``x._shape[-1] == A`` and ``y._shape[-1] == A*B``,
        ``z = x.vecmatmult(y)`` returns a :class:`GenericLazyTensor`
        such that ``z._shape[-1] == B`` which encodes, symbolically,
        the vector-matrix product of ``x`` and ``y`` along their last dimension.
        For details, please check the documentation of the KeOps operation ``"VecMacMult"`` in
        the :doc:`main reference page <../../../api/math-operations>`.
        """
        return self.binary(
            other, "VecMatMult", dimres=(other.ndim // self.ndim), dimcheck=None
        )

    def tensorprod(self, other):
        r"""
        Tensor product of vectors - a binary operation.

        If ``x._shape[-1] == A`` and ``y._shape[-1] == B``,
        ``z = x.tensorprod(y)`` returns a :class:`GenericLazyTensor`
        such that ``z._shape[-1] == A*B`` which encodes, symbolically,
        the tensor product of ``x`` and ``y`` along their last dimension.
        For details, please check the documentation of the KeOps operation ``"TensorProd"`` in
        the :doc:`main reference page <../../../api/math-operations>`.
        """
        return self.binary(
            other, "TensorProd", dimres=(other.ndim * self.ndim), dimcheck=None
        )

    def keops_tensordot(self, other, dimfa, dimfb, contfa, contfb, *args):
        """
        Tensor dot product (on KeOps internal dimensions) - a binary operation.

        :param other: a LazyTensor
        :param dimfa: tuple of int
        :param dimfb: tuple of int
        :param contfa: tuple of int listing contraction dimension of a (could be empty)
        :param contfb: tuple of int listing contraction dimension of b (could be empty)
        :param args: a tuple of int containing the graph of a permutation of the output
        :return:
        """
        # permute = tuple(range(len(dimfa) + len(dimfb) - 2 * len(contfa)))
        opt_arg = ""
        for intseq in (dimfa, dimfb, contfa, contfb) + args:
            opt_arg += "["
            if isinstance(intseq, int):
                intseq = (intseq,)  # convert to tuple
            for item in intseq:
                opt_arg += "{},".format(item)
            opt_arg = opt_arg[:-1] if len(intseq) else opt_arg  # to remove last comma
            opt_arg += "], "
        opt_arg = opt_arg[:-2]  # to remove last comma and space
        dimres = np.array(dimfa).prod() * np.array(dimfb).prod()
        dimres /= np.array(dimfa)[np.array(contfa)].prod() ** 2 if len(contfa) else 1
        return self.binary(
            other, "TensorDot", dimres=int(dimres), dimcheck=None, opt_arg=opt_arg
        )

    def grad(self, other, gradin):
        r"""
        Symbolic gradient operation.

        ``z = x.grad(v,e)`` returns a :class:`LazyTensor`
        which encodes, symbolically,
        the gradient (more precisely, the adjoint of the differential operator) of ``x``, with
        respect to variable ``v``, and applied to ``e``.
        For details, please check the documentation of the KeOps operation ``"Grad"`` in
        the :doc:`main reference page <../../../api/math-operations>`.
        """
        return self.binary(
            gradin,
            "Grad",
            dimres=other.ndim,
            dimcheck="same",
            opt_arg=other,
            opt_pos="middle",
        )

        # List of supported reductions  ============================================

    def sum(self, axis=-1, dim=None, **kwargs):
        r"""
        Summation unary operation, or Sum reduction.

        ``sum(axis, dim, **kwargs)`` will:

          - if **axis or dim = 0**, return the sum reduction of **self** over the "i" indexes.
          - if **axis or dim = 1**, return the sum reduction of **self** over the "j" indexes.
          - if **axis or dim = 2**, return a new :class:`LazyTensor` object representing the sum of the values of the vector **self**,

        Keyword Args:
          axis (integer): reduction dimension, which should be equal to the number
            of batch dimensions plus 0 (= reduction over :math:`i`),
            1 (= reduction over :math:`j`) or 2 (i.e. -1, sum along the
            dimension of the vector variable).
          dim (integer): alternative keyword for the axis parameter.
          **kwargs: optional parameters that are passed to the :meth:`reduction` method.

        """
        if dim is not None:
            axis = dim
        if axis in [-1, len(self._shape) - 1]:
            return self.unary("Sum", dimres=1)
        else:
            return self.reduction("Sum", axis=axis, **kwargs)

    def sum_reduction(self, axis=None, dim=None, **kwargs):
        r"""
        Sum reduction.

        ``sum_reduction(axis, dim, **kwargs)`` will return the sum reduction of **self**.

        Keyword Args:
          axis (integer): reduction dimension, which should be equal to the number
            of batch dimensions plus 0 (= reduction over :math:`i`),
            or 1 (= reduction over :math:`j`).
          dim (integer): alternative keyword for the axis parameter.
          **kwargs: optional parameters that are passed to the :meth:`reduction` method.

        """
        return self.reduction("Sum", axis=axis, dim=dim, **kwargs)

    def logsumexp(self, axis=None, dim=None, weight=None, **kwargs):
        r"""
        Log-Sum-Exp reduction.

        ``logsumexp(axis, dim, weight, **kwargs)`` will:

          - if **axis or dim = 0**, return the "log-sum-exp" reduction of **self** over the "i" indexes.
          - if **axis or dim = 1**, return the "log-sum-exp" reduction of **self** over the "j" indexes.

        For details, please check the documentation of the KeOps reductions ``LogSumExp`` and  ``LogSumExpWeight`` in
        the :doc:`main reference page <../../../api/math-operations>`.

        Keyword Args:
          axis (integer): reduction dimension, which should be equal to the number
            of batch dimensions plus 0 (= reduction over :math:`i`),
            or 1 (= reduction over :math:`j`).
          dim (integer): alternative keyword for the axis parameter.
          weight (:class:`LazyTensor`): optional object that specifies scalar or vector-valued weights
            in the log-sum-exp operation
          **kwargs: optional parameters that are passed to the :meth:`reduction` method.

        """
        if weight is None:
            return self.reduction("LogSumExp", axis=axis, dim=dim, **kwargs)
        else:
            return self.reduction(
                "LogSumExp", other=weight, axis=axis, dim=dim, **kwargs
            )

    def logsumexp_reduction(self, **kwargs):
        r"""
        Log-Sum-Exp reduction. Redirects to :meth:`logsumexp` method.
        """
        return self.logsumexp(**kwargs)

    def sumsoftmaxweight(self, weight, axis=None, dim=None, **kwargs):
        r"""
        Sum of weighted Soft-Max reduction.

        ``sumsoftmaxweight(weight, axis, dim, **kwargs)`` will:

          - if **axis or dim = 0**, return the "sum of weighted Soft-Max" reduction of **self** over the "i" indexes.
          - if **axis or dim = 1**, return the "sum of weighted Soft-Max" reduction of **self** over the "j" indexes.

        For details, please check the documentation of the KeOps reduction ``SumSoftMaxWeight`` in
        the :doc:`main reference page <../../../api/math-operations>`.

        Keyword Args:
          weight (:class:`LazyTensor`): object that specifies scalar or vector-valued weights.
          axis (integer): reduction dimension, which should be equal to the number
            of batch dimensions plus 0 (= reduction over :math:`i`),
            or 1 (= reduction over :math:`j`).
          dim (integer): alternative keyword for the axis parameter.
          **kwargs: optional parameters that are passed to the :meth:`reduction` method.

        """
        return self.reduction(
            "SumSoftMaxWeight", other=weight, axis=axis, dim=dim, **kwargs
        )

    def sumsoftmaxweight_reduction(self, **kwargs):
        r"""
        Sum of weighted Soft-Max reduction. Redirects to :meth:`sumsoftmaxweight` method.
        """
        return self.sumsoftmaxweight(**kwargs)

    def min(self, axis=-1, dim=None, **kwargs):
        r"""
        Minimum unary operation, or Min reduction.

        ``min(axis, dim, **kwargs)`` will:

          - if **axis or dim = 0**, return the min reduction of **self** over the "i" indexes.
          - if **axis or dim = 1**, return the min reduction of **self** over the "j" indexes.
          - if **axis or dim = 2**, return a new :class:`LazyTensor` object representing the min of the values of the vector **self**,

        Keyword Args:
          axis (integer): reduction dimension, which should be equal to the number
            of batch dimensions plus 0 (= reduction over :math:`i`),
            1 (= reduction over :math:`j`) or 2 (i.e. -1, min along the
            dimension of the vector variable).
          dim (integer): alternative keyword for the axis parameter.
          **kwargs: optional parameters that are passed to the :meth:`reduction` method.

        """
        if dim is not None:
            axis = dim
        if axis in [-1, len(self._shape) - 1]:
            return self.unary("Min", dimres=1)
        else:
            return self.reduction("Min", axis=axis, **kwargs)

    def min_reduction(self, axis=None, dim=None, **kwargs):
        r"""
        Min reduction.

        ``min_reduction(axis, dim, **kwargs)`` will return the min reduction of **self**.

        Keyword Args:
          axis (integer): reduction dimension, which should be equal to the number
            of batch dimensions plus 0 (= reduction over :math:`i`),
            or 1 (= reduction over :math:`j`).
          dim (integer): alternative keyword for the axis parameter.
          **kwargs: optional parameters that are passed to the :meth:`reduction` method.

        """
        return self.reduction("Min", axis=axis, dim=dim, **kwargs)

    def __min__(self, **kwargs):
        r"""
        Minimum unary operation, or Min reduction. Redirects to :meth:`min` method.
        """
        return self.min(**kwargs)

    def argmin(self, axis=-1, dim=None, **kwargs):
        r"""
        ArgMin unary operation, or ArgMin reduction.

        ``argmin(axis, dim, **kwargs)`` will:

          - if **axis or dim = 0**, return the argmin reduction of **self** over the "i" indexes.
          - if **axis or dim = 1**, return the argmin reduction of **self** over the "j" indexes.
          - if **axis or dim = 2**, return a new :class:`LazyTensor` object representing the argmin of the values of the vector **self**,

        Keyword Args:
          axis (integer): reduction dimension, which should be equal to the number
            of batch dimensions plus 0 (= reduction over :math:`i`),
            1 (= reduction over :math:`j`) or 2 (i.e. -1, argmin along the
            dimension of the vector variable).
          dim (integer): alternative keyword for the axis parameter.
          **kwargs: optional parameters that are passed to the :meth:`reduction` method.

        """
        if dim is not None:
            axis = dim
        if axis in [-1, len(self._shape) - 1]:
            return self.unary("ArgMin", dimres=1)
        else:
            return self.reduction("ArgMin", axis=axis, **kwargs)

    def argmin_reduction(self, axis=None, dim=None, **kwargs):
        r"""
        ArgMin reduction.

        ``argmin_reduction(axis, dim, **kwargs)`` will return the argmin reduction of **self**.

        Keyword Args:
          axis (integer): reduction dimension, which should be equal to the number
            of batch dimensions plus 0 (= reduction over :math:`i`),
            or 1 (= reduction over :math:`j`).
          dim (integer): alternative keyword for the axis parameter.
          **kwargs: optional parameters that are passed to the :meth:`reduction` method.

        """
        return self.reduction("ArgMin", axis=axis, dim=dim, **kwargs)

    def min_argmin(self, axis=None, dim=None, **kwargs):
        r"""
        Min-ArgMin reduction.

        ``min_argmin(axis, dim, **kwargs)`` will:

          - if **axis or dim = 0**, return the minimal values and its indices of **self** over the "i" indexes.
          - if **axis or dim = 1**, return the minimal values and its indices of **self** over the "j" indexes.

        Keyword Args:
          axis (integer): reduction dimension, which should be equal to the number
            of batch dimensions plus 0 (= reduction over :math:`i`),
            or 1 (= reduction over :math:`j`).
          dim (integer): alternative keyword for the axis parameter.
          **kwargs: optional parameters that are passed to the :meth:`reduction` method.

        """
        return self.reduction("Min_ArgMin", axis=axis, dim=dim, **kwargs)

    def min_argmin_reduction(self, **kwargs):
        r"""
        Min-ArgMin reduction. Redirects to :meth:`min_argmin` method.
        """
        return self.min_argmin(**kwargs)

    def max(self, axis=-1, dim=None, **kwargs):
        r"""
        Miaximum unary operation, or Max reduction.

        ``max(axis, dim, **kwargs)`` will:

          - if **axis or dim = 0**, return the max reduction of **self** over the "i" indexes.
          - if **axis or dim = 1**, return the max reduction of **self** over the "j" indexes.
          - if **axis or dim = 2**, return a new :class:`LazyTensor` object representing the max of the values of the vector **self**,

        Keyword Args:
          axis (integer): reduction dimension, which should be equal to the number
            of batch dimensions plus 0 (= reduction over :math:`i`),
            1 (= reduction over :math:`j`) or 2 (i.e. -1, max along the
            dimension of the vector variable).
          dim (integer): alternative keyword for the axis parameter.
          **kwargs: optional parameters that are passed to the :meth:`reduction` method.

        """
        if dim is not None:
            axis = dim
        if axis in [-1, len(self._shape) - 1]:
            return self.unary("Max", dimres=1)
        else:
            return self.reduction("Max", axis=axis, **kwargs)

    def max_reduction(self, axis=None, dim=None, **kwargs):
        r"""
        Max reduction.

        ``max_reduction(axis, dim, **kwargs)`` will return the max reduction of **self**.

        Keyword Args:
          axis (integer): reduction dimension, which should be equal to the number
            of batch dimensions plus 0 (= reduction over :math:`i`),
            or 1 (= reduction over :math:`j`).
          dim (integer): alternative keyword for the axis parameter.
          **kwargs: optional parameters that are passed to the :meth:`reduction` method.

        """
        return self.reduction("Max", axis=axis, dim=dim, **kwargs)

    def __max__(self, **kwargs):
        r"""
        Maximum unary operation, or Max reduction. Redirects to :meth:`max` method.
        """
        return self.max(**kwargs)

    def argmax(self, axis=-1, dim=None, **kwargs):
        r"""
        ArgMax unary operation, or ArgMax reduction.

        ``argmax(axis, dim, **kwargs)`` will:

          - if **axis or dim = 0**, return the argmax reduction of **self** over the "i" indexes.
          - if **axis or dim = 1**, return the argmax reduction of **self** over the "j" indexes.
          - if **axis or dim = 2**, return a new :class:`LazyTensor` object representing the argmax of the values of the vector **self**,

        Keyword Args:
          axis (integer): reduction dimension, which should be equal to the number
            of batch dimensions plus 0 (= reduction over :math:`i`),
            1 (= reduction over :math:`j`) or 2 (i.e. -1, argmax along the
            dimension of the vector variable).
          dim (integer): alternative keyword for the axis parameter.
          **kwargs: optional parameters that are passed to the :meth:`reduction` method.

        """
        if dim is not None:
            axis = dim
        if axis in [-1, len(self._shape) - 1]:
            return self.unary("ArgMax", dimres=1)
        else:
            return self.reduction("ArgMax", axis=axis, **kwargs)

    def argmax_reduction(self, axis=None, dim=None, **kwargs):
        r"""
        ArgMax reduction.

        ``argmax_reduction(axis, dim, **kwargs)`` will return the argmax reduction of **self**.

        Keyword Args:
          axis (integer): reduction dimension, which should be equal to the number
            of batch dimensions plus 0 (= reduction over :math:`i`),
            or 1 (= reduction over :math:`j`).
          dim (integer): alternative keyword for the axis parameter.
          **kwargs: optional parameters that are passed to the :meth:`reduction` method.

        """
        return self.reduction("ArgMax", axis=axis, dim=dim, **kwargs)

    def max_argmax(self, axis=None, dim=None, **kwargs):
        r"""
        Max-ArgMax reduction.

        ``max_argmax(axis, dim, **kwargs)`` will:

          - if **axis or dim = 0**, return the maximal values and its indices of **self** over the "i" indexes.
          - if **axis or dim = 1**, return the maximal values and its indices of **self** over the "j" indexes.

        Keyword Args:
          axis (integer): reduction dimension, which should be equal to the number
            of batch dimensions plus 0 (= reduction over :math:`i`),
            or 1 (= reduction over :math:`j`).
          dim (integer): alternative keyword for the axis parameter.
          **kwargs: optional parameters that are passed to the :meth:`reduction` method.

        """
        return self.reduction("Max_ArgMax", axis=axis, dim=dim, **kwargs)

    def max_argmax_reduction(self, **kwargs):
        r"""
        Max-ArgMax reduction. Redirects to :meth:`max_argmax` method.
        """
        return self.max_argmax(**kwargs)

    def Kmin(self, K, axis=None, dim=None, **kwargs):
        r"""
        K-Min reduction.

        ``Kmin(K, axis, dim, **kwargs)`` will:

          - if **axis or dim = 0**, return the K minimal values of **self** over the "i" indexes.
          - if **axis or dim = 1**, return the K minimal values of **self** over the "j" indexes.

        Keyword Args:
          K (integer): number of minimal values required
          axis (integer): reduction dimension, which should be equal to the number
            of batch dimensions plus 0 (= reduction over :math:`i`),
            or 1 (= reduction over :math:`j`).
          dim (integer): alternative keyword for the axis parameter.
          **kwargs: optional parameters that are passed to the :meth:`reduction` method.

        """
        return self.reduction("KMin", opt_arg=K, axis=axis, dim=dim, **kwargs)

    def Kmin_reduction(self, **kwargs):
        r"""
        Kmin reduction. Redirects to :meth:`Kmin` method.
        """
        return self.Kmin(**kwargs)

    def argKmin(self, K, axis=None, dim=None, **kwargs):
        r"""
        argKmin reduction.

        ``argKmin(K, axis, dim, **kwargs)`` will:

          - if **axis or dim = 0**, return the indices of the K minimal values of **self** over the "i" indexes.
          - if **axis or dim = 1**, return the indices of the K minimal values of **self** over the "j" indexes.

        Keyword Args:
          K (integer): number of minimal values required
          axis (integer): reduction dimension, which should be equal to the number
            of batch dimensions plus 0 (= reduction over :math:`i`),
            or 1 (= reduction over :math:`j`).
          dim (integer): alternative keyword for the axis parameter.
          **kwargs: optional parameters that are passed to the :meth:`reduction` method.

        """
        return self.reduction("ArgKMin", opt_arg=K, axis=axis, dim=dim, **kwargs)

    def argKmin_reduction(self, **kwargs):
        r"""
        argKmin reduction. Redirects to :meth:`argKmin` method.
        """
        return self.argKmin(**kwargs)

    def Kmin_argKmin(self, K, axis=None, dim=None, **kwargs):
        r"""
        K-Min-argK-min reduction.

        ``Kmin_argKmin(K, axis, dim, **kwargs)`` will:

          - if **axis or dim = 0**, return the K minimal values and its indices of **self** over the "i" indexes.
          - if **axis or dim = 1**, return the K minimal values and its indices of **self** over the "j" indexes.

        Keyword Args:
          K (integer): number of minimal values required
          axis (integer): reduction dimension, which should be equal to the number
            of batch dimensions plus 0 (= reduction over :math:`i`),
            or 1 (= reduction over :math:`j`).
          dim (integer): alternative keyword for the axis parameter.
          **kwargs: optional parameters that are passed to the :meth:`reduction` method.

        """
        return self.reduction("KMin_ArgKMin", opt_arg=K, axis=axis, dim=dim, **kwargs)

    def Kmin_argKmin_reduction(self, **kwargs):
        r"""
        Kmin_argKmin reduction. Redirects to :meth:`Kmin_argKmin` method.
        """
        return self.Kmin_argKmin(**kwargs)

    # LazyTensors as linear operators  =========================================

    def __matmul__(self, v, **kwargs):
        r"""
        Matrix-vector or Matrix-matrix product, supporting batch dimensions.

        If ``K`` is a :class:`LazyTensor` whose trailing dimension ``K._shape[-1]`` is equal to 1,
        we can understand it as a linear operator and apply it to arbitrary NumPy arrays
        or PyTorch Tensors. Assuming that ``v`` is a 1D (resp. ND) tensor such that
        ``K.shape[-1] == v.shape[-1]`` (resp. ``v.shape[-2]``), ``K @ v`` denotes the matrix-vector (resp. matrix-matrix)
        product between the two objects, encoded as a vanilla NumPy or PyTorch 1D (resp. ND) tensor.

        Example:
            >>> x, y = torch.randn(1000, 3), torch.randn(2000, 3)
            >>> x_i, y_j = LazyTensor( x[:,None,:] ), LazyTensor( y[None,:,:] )
            >>> K = (- ((x_i - y_j)**2).sum(2) ).exp()  # Symbolic (1000,2000,1) Gaussian kernel matrix
            >>> v = torch.rand(2000, 2)
            >>> print( (K @ v).shape )
            ... torch.Size([1000, 2])
        """
        if self._shape[-1] != 1:
            raise ValueError(
                "The 'K @ v' syntax is only supported for LazyTensors "
                + "'K' whose trailing dimension is equal to 1. Here, K.shape = {}.".format(
                    self.shape
                )
            )

        if len(v.shape) == 1:
            newdims = (1, v.shape[0], 1)
        else:
            newdims = v.shape[:-2] + (1,) + v.shape[-2:]

        v_ = self.lt_constructor(self.tools.view(v, newdims))
        Kv = self * v_  # Supports broadcasting
        Kv = Kv.sum(Kv.dim() - 2, **kwargs)  # Matrix-vector or Matrix-matrix product

        # Expected behavior: if v is a vector, so should K @ v.
        return self.tools.view(Kv, -1) if len(v.shape) == 1 else Kv

    def t(self):
        r"""
        Matrix transposition, permuting the axes of :math:`i`- and :math:`j`-variables.

        For instance, if ``K`` is a LazyTensor of shape ``(B,M,N,D)``,
        ``K.t()`` returns a symbolic copy of ``K`` whose axes 1 and 2 have
        been switched with each other: ``K.t().shape == (B,N,M,D)``.

        Example:
            >>> x, y = torch.randn(1000, 3), torch.randn(2000, 3)
            >>> x_i, y_j = LazyTensor( x[:,None,:] ), LazyTensor( y[None,:,:] )
            >>> K  = (- ((    x_i     -      y_j   )**2).sum(2) ).exp()  # Symbolic (1000,2000) Gaussian kernel matrix
            >>> K_ = (- ((x[:,None,:] - y[None,:,:])**2).sum(2) ).exp()  # Explicit (1000,2000) Gaussian kernel matrix
            >>> w  = torch.rand(1000, 2)
            >>> print( (K.t() @ w - K_.t() @ w).abs().mean() )
            ... tensor(1.7185e-05)
        """

        res = copy.copy(self)
        res.ni, res.nj = res.nj, res.ni  # Switch the "M" and "N" dimensions
        res.ranges = res.tools.swap_axes(res.ranges)

        if res.axis == 0:
            res.axis = 1
        elif res.axis == 1:
            res.axis = 0

        if res.formula is not None:  # Switch variables with CAT=0 and CAT=1
            res.formula = re.sub(
                r"(Var|VarSymb)\((\d+),(\d+),0\)", r"\1(\2,\3,i)", res.formula
            )
            res.formula = re.sub(
                r"(Var|VarSymb)\((\d+),(\d+),1\)", r"\1(\2,\3,0)", res.formula
            )
            res.formula = re.sub(
                r"(Var|VarSymb)\((\d+),(\d+),i\)", r"\1(\2,\3,1)", res.formula
            )

        if res.formula2 is not None:  # Switch variables with CAT=0 and CAT=1
            res.formula2 = re.sub(
                r"(Var|VarSymb)\((\d+),(\d+),0\)", r"\1(\2,\3,i)", res.formula2
            )
            res.formula2 = re.sub(
                r"(Var|VarSymb)\((\d+),(\d+),1\)", r"\1(\2,\3,0)", res.formula2
            )
            res.formula2 = re.sub(
                r"(Var|VarSymb)\((\d+),(\d+),i\)", r"\1(\2,\3,1)", res.formula2
            )
        # we need also to make copies of references for all variables in the formula
        # that were switched
        newvars = []
        for x in self.variables:
            if type(x) == list:
                # here we are dealing with a parameter variable, so no need to do any copy
                newvars.append(x)
            else:
                y = self.tools.view(x, x.shape)
                newvars.append(y)
                # now we replace all occurrences of old ids by new ids in formulas
                if res.formula is not None:
                    res.formula = re.sub(
                        r"(Var|VarSymb)\({},(\d+),(\d+)\)".format(id(x)),
                        r"\1({},\2,\3)".format(id(y)),
                        res.formula,
                    )
                if res.formula2 is not None:
                    res.formula2 = re.sub(
                        r"(Var|VarSymb)\({},(\d+),(\d+)\)".format(id(x)),
                        r"\1({},\2,\3)".format(id(y)),
                        res.formula2,
                    )
        res.variables = tuple(newvars)
        return res

    @property
    def T(self):
        r"""
        Numpy-friendly alias for the matrix transpose ``self.t()``.
        """
        return self.t()

    def matvec(self, v):
        r"""
        Alias for the matrix-vector product, added for compatibility with :mod:`scipy.sparse.linalg`.

        If ``K`` is a :class:`LazyTensor` whose trailing dimension ``K._shape[-1]`` is equal to 1,
        we can understand it as a linear operator and wrap it into a
        :mod:`scipy.sparse.linalg.LinearOperator` object, thus getting access
        to robust solvers and spectral routines.

        Example:
            >>> import numpy as np
            >>> x = np.random.randn(1000,3)
            >>> x_i, x_j = LazyTensor( x[:,None,:] ), LazyTensor( x[None,:,:] )
            >>> K_xx = (- ((x_i - x_j)**2).sum(2) ).exp()  # Symbolic (1000,1000) Gaussian kernel matrix
            >>> from scipy.sparse.linalg import eigsh, aslinearoperator
            >>> eigenvalues, eigenvectors = eigsh( aslinearoperator( K_xx ), k=5 )
            >>> print(eigenvalues)
            ... [ 35.5074527   59.01096445  61.35075268  69.34038814 123.77540277]
            >>> print( eigenvectors.shape)
            ... (1000, 5)
        """
        return self @ v

    def rmatvec(self, v):
        r"""
        Alias for the transposed matrix-vector product, added for compatibility with :mod:`scipy.sparse.linalg`.

        See :meth:`matvec` for further reference.
        """
        return self.T @ v

    def real2complex(self):
        r"""
        Element-wise "real 2 complex" operation - a unary operation.

        ``x.real2complex()`` returns a :class:`ComplexLazyTensor` that encodes, symbolically,
        the same tensor as ``x``, but seen as complex-valued (with zero imaginary part for each coefficient)
        """
        return self.unary("Real2Complex", dimres=2 * self._shape[-1], is_complex=True)

    def imag2complex(self):
        r"""
        Element-wise "imag 2 complex" operation - a unary operation.

        ``x.real2complex()`` returns a :class:`ComplexLazyTensor` that encodes, symbolically,
        the multiplication of ``1j`` with ``x``.
        """
        return self.unary("Imag2Complex", dimres=2 * self._shape[-1], is_complex=True)

    def exp1j(self):
        r"""
        Element-wise "complex exponential of 1j x" operation - a unary operation.

        ``x.exp1j()`` returns a :class:`ComplexLazyTensor` that encodes, symbolically,
        the complex exponential of ``1j*x``.
        """
        return self.unary("ComplexExp1j", dimres=2 * self._shape[-1], is_complex=True)


class ComplexGenericLazyTensor(GenericLazyTensor):
    r"""Extension of the LazyTensor class for complex operations."""

    def __init__(self, x=None, axis=None):
        r"""Creates a KeOps symbolic variable of complex dtype."""
        self.get_tools()
        if type(x) == complex:
            x = [x]
        if type(x) == list:
            x_ = [None] * (2 * len(x))
            for i in range(len(x)):
                x_[2 * i] = x[i].real
                x_[2 * i + 1] = x[i].imag
                x = x_
        elif self.tools.is_tensor(x):
            x = self.tools.view_as_real(x)
        super().__init__(x=x, axis=axis)
        self.is_complex = True

    def __call__(self, *args, **kwargs):
        res = super().__call__(*args, **kwargs)
        return self.tools.view_as_complex(res)

    @property
    def dtype(self):
        if self._dtype == "float32":
            return "complex64"
        elif self._dtype == "float64":
            return "complex128"

    @property
    def shape(self):
        r"""returns the shape of the complex LazyTensor."""
        s = super()._shape
        s = s[:-1] + (s[-1] // 2,)
        if s[-1] == 1:
            return s[:-1]
        else:
            return s

    # List of supported operations  ============================================

    @property
    def real(self):
        r"""
        Element-wise real part of complex - a unary operation.

        ``z.real`` returns a :class:`LazyTensor` that encodes, symbolically,
        the element-wise real part of ``z``.
        """
        return self.unary("ComplexReal", dimres=self._shape[-1] // 2, is_complex=False)

    @property
    def imag(self):
        r"""
        Element-wise imaginary part of complex - a unary operation.

        ``z.imag`` returns a :class:`LazyTensor` that encodes, symbolically,
        the element-wise imaginary part of ``z``.
        """
        return self.unary("ComplexImag", dimres=self._shape[-1] // 2, is_complex=False)

    def angle(self):
        r"""
        Element-wise angle (or argument) of complex - a unary operation.

        ``z.angle()`` returns a :class:`LazyTensor` that encodes, symbolically,
        the element-wise angle of ``z``.
        """
        return self.unary("ComplexAngle", dimres=self._shape[-1] // 2, is_complex=False)

    def conj(self):
        r"""
        Element-wise complex conjugate - a unary operation.

        ``z.conj()`` returns a :class:`ComplexLazyTensor` that encodes, symbolically,
        the element-wise complex conjugate of ``z``.
        """
        return self.unary("Conj", dimres=self._shape[-1], is_complex=True)

    def sum(self, axis=-1, dim=None, **kwargs):
        if dim is not None:
            axis = dim
        if axis in [-1, len(self._shape) - 1]:
            return self.unary("ComplexSum", dimres=2, is_complex=True)
        else:
            return self.reduction("Sum", axis=axis, **kwargs)

    def __abs__(self):
        r"""
        Element-wise absolute value (or modulus) of complex - a unary operation.

        ``z.abs()`` returns a :class:`LazyTensor` that encodes, symbolically,
        the element-wise absolute value of ``z``.
        """
        return self.unary("ComplexAbs", dimres=self._shape[-1] // 2, is_complex=False)

    def exp(self):
        r"""
        Element-wise complex exponential - a unary operation.

        ``z.exp()`` returns a :class:`ComplexLazyTensor` that encodes, symbolically,
        the element-wise complex exponential of ``z``.
        """
        return self.unary("ComplexExp", dimres=self._shape[-1], is_complex=True)

    def mulop(self, other, **kwargs):
        if other._shape[-1] == 1:
            return other.binary(self, "ComplexRealScal", **kwargs, is_complex=True)
        elif not is_complex_lazytensor(other):
            return self.mulop(other.real2complex())
        else:
            return self.binary(
                other, "ComplexMult", **kwargs, is_complex=True, dimcheck=None
            )

    def addop(self, other, **kwargs):
        if not is_complex_lazytensor(other):
            return self.addop(other.real2complex())
        else:
            return self.binary(
                other, "ComplexAdd", **kwargs, is_complex=True, dimcheck=None
            )

    def subop(self, other, **kwargs):
        if not is_complex_lazytensor(other):
            return self.subop(other.real2complex())
        else:
            return self.binary(
                other, "ComplexSubtract", **kwargs, is_complex=True, dimcheck=None
            )

    def divop(self, other, **kwargs):
        if not is_complex_lazytensor(other):
            return self.divop(other.real2complex())
        else:
            return self.binary(
                other, "ComplexDivide", **kwargs, is_complex=True, dimcheck=None
            )

    def real2complex(self):
        raise ValueError("real2complex cannot be applied to a complex LazyTensor.")

    def imag2complex(self):
        raise ValueError("imag2complex cannot be applied to a complex LazyTensor.")

    def exp1j(self):
        raise ValueError("exp1j cannot be applied to a complex LazyTensor.")

    def __call__(self, *args, **kwargs):
        res = super().__call__(*args, **kwargs)
        return self.tools.view_as_complex(res)
