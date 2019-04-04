from pykeops.common.keops_io import load_keops
from pykeops.common.get_options import get_tag_backend
from pykeops.common.parse_type import get_sizes, complete_aliases
from pykeops.common.utils import axis2cat
from pykeops.numpy import default_dtype
from pykeops.common.operations import preprocess, postprocess


class Genred():
    r"""
        Creates a new generic operation.

        This is KeOps' main function, whose usage is documented in
        the :doc:`user-guide <../../Genred>`,
        the :doc:`gallery of examples <../../../_auto_examples/index>`
        and the :doc:`high-level tutorials <../../../_auto_tutorials/index>`.
        Taking as input a handful of strings and integers that specify
        a custom Map-Reduce operation, it returns a C++ wrapper
        that can be called just like any other NumPy function.

        Warning:
            Even for variables of size 1 (e.g. :math:`a_i\in\mathbb{R}`
            for :math:`i\in[0,M)`), KeOps expects inputs to be formatted
            as 2d Tensors of size ``(M,dim)``. In practice,
            ``a.view(-1,1)`` should be used to turn a vector of weights
            into a *list of scalar values*.

        Warning:
            As of today, vector-valued formulas are only supported by
            the ``"Sum"`` reduction. All the
            :ref:`other reductions <part.reduction>` expect
            **formula** to be scalar-valued.

        Note:
            :func:`Genred` relies on CUDA kernels that are compiled on-the-fly,
            and stored in ``pykeops.build_folder`` as ".dll" or ".so" files for later use.

        Note:
            On top of the **Sum** and **LogSumExp** reductions, KeOps
            supports
            :ref:`variants of the ArgKMin reduction <part.reduction>`
            that can be used
            to implement k-nearest neighbor search.
            These routines return indices encoded as **floating point numbers**, and
            produce no gradient. Fortunately though, you can simply
            turn them into ``LongTensors`` and use them to index
            your arrays, as showcased in the documentation
            of :func:`generic_argmin`, :func:`generic_argkmin` and in the
            :doc:`K-means tutorial <../../../_auto_tutorials/kmeans/plot_kmeans_numpy>`.


        Args:
            formula (string): The scalar- or vector-valued expression
                that should be computed and reduced.
                The correct syntax is described in the :doc:`documentation <../../Genred>`,
                using appropriate :doc:`mathematical operations <../../../api/math-operations>`.
            aliases (list of strings): A list of identifiers of the form ``"AL = TYPE(DIM)"``
                that specify the categories and dimensions of the input variables. Here:

                  - ``AL`` is an alphanumerical alias, used in the **formula**.
                  - ``TYPE`` is a *category*. One of:

                    - ``Vi``: indexation by :math:`i` along axis 0.
                    - ``Vj``: indexation by :math:`j` along axis 1.
                    - ``Pm``: no indexation, the input tensor is a *vector* and not a 2d array.

                  - ``DIM`` is an integer, the dimension of the current variable.

                As described below, :meth:`__call__` will expect as input Tensors whose
                shape are compatible with **aliases**.

        Keyword Args:
            reduction_op (string, default = ``"Sum"``): Specifies the reduction
                operation that is applied to reduce the values
                of ``formula(x_i, y_j, ...)`` along axis 0 or axis 1.
                The supported values are one of :ref:`part.reduction`

            axis (int, default = 0): Specifies the dimension of the "kernel matrix" that is reduced by our routine.
                The supported values are:

                  - **axis** = 0: reduction with respect to :math:`i`, outputs a ``Vj`` or ":math:`j`" variable.
                  - **axis** = 1: reduction with respect to :math:`j`, outputs a ``Vi`` or ":math:`i`" variable.

            dtype (string, default = ``"float32"``): Specifies the numerical ``dtype`` of the input and output arrays.
                The supported values are:

                  - **dtype** = ``"float32"`` or ``"float"``.
                  - **dtype** = ``"float64"`` or ``"double"``.

            opt_arg (int, default = None): If **reduction_op** is in ``["KMin", "ArgKMin", "KMinArgKMin"]``,
                this argument allows you to specify the number ``K`` of neighbors to consider.



        **To apply the routine on arbitrary NumPy arrays:**


        Args:
            *args (2d arrays (variables ``Vi(..)``, ``Vj(..)``) and 1d arrays (parameters ``Pm(..)``)): The input numerical arrays,
                which should all have the same ``dtype``, be **contiguous** and be stored on
                the **same device**. KeOps expects one array per alias,
                with the following compatibility rules:

                    - All ``Vi(Dim_k)`` variables are encoded as **2d-arrays** with ``Dim_k`` columns and the same number of lines :math:`M`.
                    - All ``Vj(Dim_k)`` variables are encoded as **2d-arrays** with ``Dim_k`` columns and the same number of lines :math:`N`.
                    - All ``Pm(Dim_k)`` variables are encoded as **1d-arrays** (vectors) of size ``Dim_k``.

        Keyword Args:
            backend (string): Specifies the map-reduce scheme.
                The supported values are:

                    - ``"auto"`` (default): let KeOps decide which backend is best suited to your data, based on the tensors' shapes. ``"GPU_1D"`` will be chosen in most cases.
                    - ``"CPU"``: use a simple C++ ``for`` loop on a single CPU core.
                    - ``"GPU_1D"``: use a `simple multithreading scheme <https://gitlab.com/bcharlier/keops/blob/master/keops/core/GpuConv1D.cu>`_ on the GPU - basically, one thread per value of the output index.
                    - ``"GPU_2D"``: use a more sophisticated `2D parallelization scheme <https://gitlab.com/bcharlier/keops/blob/master/keops/core/GpuConv2D.cu>`_ on the GPU.
                    - ``"GPU"``: let KeOps decide which one of the ``"GPU_1D"`` or the ``"GPU_2D"`` scheme will run faster on the given input.

            device_id (int, default=-1): Specifies the GPU that should be used
                to perform the computation; a negative value lets your system
                choose the default GPU. This parameter is only useful if your
                system has access to several GPUs.

            ranges (6-uple of integer arrays, None by default):
                Ranges of integers that specify a
                :doc:`block-sparse reduction scheme <../../sparsity>`
                with *Mc clusters along axis 0* and *Nc clusters along axis 1*.
                If None (default), we simply loop over all indices
                :math:`i\in[0,M)` and :math:`j\in[0,N)`.

                **The first three ranges** will be used if **axis** = 1
                (reduction along the axis of ":math:`j` variables"),
                and to compute gradients with respect to ``Vi(..)`` variables:

                    - ``ranges_i``, (Mc,2) integer array - slice indices
                      :math:`[\operatorname{start}^I_k,\operatorname{end}^I_k)` in :math:`[0,M]`
                      that specify our Mc blocks along the axis 0
                      of ":math:`i` variables".
                    - ``slices_i``, (Mc,2) integer array - slice indices
                      :math:`[\operatorname{start}^S_k,\operatorname{end}^S_k)`
                      that specify Mc ranges in ``redranges_j``.
                    - ``redranges_j``, (Mcc,2) integer array - slice indices
                      :math:`[\operatorname{start}^J_l,\operatorname{end}^J_l)` in :math:`[0,N]`
                      that specify reduction ranges along the axis 1
                      of ":math:`j` variables".

                If **axis** = 1,
                these integer arrays allow us to say
                that ``for k in range(Mc)``, the output values for
                indices ``i in range( ranges_i[k,0], ranges_i[k,1] )``
                should be computed using a Map-Reduce scheme over
                indices ``j in Union( range( redranges_j[l, 0], redranges_j[l, 1] ))``
                for ``l in range( slices_i[k,0], slices_i[k,1] )``.

                **Likewise, the last three ranges** will be used if **axis** = 0
                (reduction along the axis of ":math:`i` variables"),
                and to compute gradients with respect to ``Vj(..)`` variables:

                    - ``ranges_j``, (Nc,2) integer array - slice indices
                      :math:`[\operatorname{start}^J_k,\operatorname{end}^J_k)` in :math:`[0,N]`
                      that specify our Nc blocks along the axis 1
                      of ":math:`j` variables".
                    - ``slices_j``, (Nc,2) integer array - slice indices
                      :math:`[\operatorname{start}^S_k,\operatorname{end}^S_k)`
                      that specify Nc ranges in ``redranges_i``.
                    - ``redranges_i``, (Ncc,2) integer array - slice indices
                      :math:`[\operatorname{start}^I_l,\operatorname{end}^I_l)` in :math:`[0,M]`
                      that specify reduction ranges along the axis 0
                      of ":math:`i` variables".

                If **axis** = 0,
                these integer arrays allow us to say
                that ``for k in range(Nc)``, the output values for
                indices ``j in range( ranges_j[k,0], ranges_j[k,1] )``
                should be computed using a Map-Reduce scheme over
                indices ``i in Union( range( redranges_i[l, 0], redranges_i[l, 1] ))``
                for ``l in range( slices_j[k,0], slices_j[k,1] )``.

        Returns:
            (M,D) or (N,D) array:

            The output of the reduction,
            a **2d-tensor** with :math:`M` or :math:`N` lines (if **axis** = 1
            or **axis** = 0, respectively) and a number of columns
            that is inferred from the **formula**.



        Example:
            >>> my_conv = Genred('Exp(-SqNorm2(x - y))',  # formula
            ...                  ['x = Vi(3)',            # 1st input: dim-3 vector per line
            ...                   'y = Vj(3)'],           # 2nd input: dim-3 vector per column
            ...                  reduction_op='Sum',      # we also support LogSumExp, Min, etc.
            ...                  axis=1)                  # reduce along the lines of the kernel matrix
            >>> # Apply it to 2d arrays x and y with 3 columns and a (huge) number of lines
            >>> x = np.random.randn(1000000, 3)
            >>> y = np.random.randn(2000000, 3)
            >>> a = my_conv(x, y)  # a_i = sum_j exp(-|x_i-y_j|^2)
            >>> print(a.shape)
            [1000000, 1]
        """
    
    def __init__(self, formula, aliases, reduction_op='Sum', axis=0, dtype=default_dtype, opt_arg=None,
                 formula2=None, cuda_type=None):
        if cuda_type:
            # cuda_type is just old keyword for dtype, so this is just a trick to keep backward compatibility
            dtype = cuda_type 
        self.reduction_op = reduction_op
        reduction_op_internal, formula2 = preprocess(reduction_op, formula2)
        
        str_opt_arg = ',' + str(opt_arg) if opt_arg else ''
        str_formula2 = ',' + formula2 if formula2 else ''
        
        self.formula = reduction_op_internal + '_Reduction(' + formula + str_opt_arg + ',' + str(
            axis2cat(axis)) + str_formula2 + ')'
        self.aliases = complete_aliases(self.formula, aliases)
        self.dtype = dtype
        self.myconv = load_keops(self.formula,  self.aliases,  self.dtype, 'numpy')
        self.axis = axis
        self.opt_arg = opt_arg

    def __call__(self, *args, backend='auto', device_id=-1, ranges=None):
        # Get tags
        tagCpuGpu, tag1D2D, _ = get_tag_backend(backend, args)
        nx, ny = get_sizes(self.aliases, *args)
        if ranges is None : ranges = () # To keep the same type

        out = self.myconv.genred_numpy(nx, ny, tagCpuGpu, tag1D2D, 0, device_id, ranges, *args)

        nout = nx if self.axis==1 else ny
        return postprocess(out, "numpy", self.reduction_op, nout, self.opt_arg, self.dtype)
