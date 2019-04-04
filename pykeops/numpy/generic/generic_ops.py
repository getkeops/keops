from pykeops.common.parse_type import get_type
from pykeops.common.utils import cat2axis
from pykeops.numpy import Genred


def generic_sum(formula, output, *aliases, **kwargs):
    """Alias for :class:`pykeops.numpy.Genred` with a "Sum" reduction.

    Args:
        formula (string): Symbolic KeOps expression, as in :class:`pykeops.numpy.Genred`.
        output (string): An identifier of the form ``"AL = TYPE(DIM)"`` 
            that specifies the category and dimension of the output variable. Here:

              - ``AL`` is a dummy alphanumerical name.
              - ``TYPE`` is a *category*. One of:

                - ``Vi``: indexation by :math:`i` along axis 0; reduction is performed along axis 1.
                - ``Vj``: indexation by :math:`j` along axis 1; reduction is performed along axis 0.

              - ``DIM`` is an integer, the dimension of the output variable; it should be compatible with **formula**.
        *aliases (strings): List of identifiers, as in :class:`pykeops.numpy.Genred`.

    Keyword Args:
        dtype (string, default = ``"float64"``): Specifies the numerical **dtype** of the input and output arrays. 
            The supported values are:

              - **dtype** = ``"float32"``,
              - **dtype** = ``"float64"``.

    Returns:
        A generic reduction that can be called on arbitrary
        NumPy arrays, as documented in :class:`pykeops.numpy.Genred`.

    Example:
        >>> my_conv = generic_sum(       # Custom Kernel Density Estimator
        ...     'Exp(-SqNorm2(x - y))',  # Formula
        ...     'a = Vi(1)',             # Output: 1 scalar per line
        ...     'x = Vi(3)',             # 1st input: dim-3 vector per line
        ...     'y = Vj(3)')             # 2nd input: dim-3 vector per line
        >>> # Apply it to 2d arrays x and y with 3 columns and a (huge) number of lines
        >>> x = np.random.randn(1000000, 3)
        >>> y = np.random.randn(2000000, 3)
        >>> a = my_conv(x, y)  # a_i = sum_j exp(-|x_i-y_j|^2)
        >>> print(a.shape)
        (1000000, 1)
    """
    _, cat, _, _ = get_type(output)
    return Genred(formula, list(aliases), reduction_op='Sum', axis=cat2axis(cat), **kwargs)


def generic_logsumexp(formula, output, *aliases, **kwargs):
    """Alias for :class:`pykeops.numpy.Genred` with a "LogSumExp" reduction.

    Args:
        formula (string): Scalar-valued symbolic KeOps expression, as in :class:`pykeops.numpy.Genred`.
        output (string): An identifier of the form ``"AL = TYPE(1)"`` 
            that specifies the category and dimension of the output variable. Here:

              - ``AL`` is a dummy alphanumerical name.
              - ``TYPE`` is a *category*. One of:

                - ``Vi``: indexation by :math:`i` along axis 0; reduction is performed along axis 1.
                - ``Vj``: indexation by :math:`j` along axis 1; reduction is performed along axis 0.

        *aliases (strings): List of identifiers, as in :class:`pykeops.numpy.Genred`.

    Keyword Args:
        dtype (string, default = ``"float64"``): Specifies the numerical **dtype** of the input and output arrays. 
            The supported values are:

              - **dtype** = ``"float32"``,
              - **dtype** = ``"float64"``.

    Returns:
        A generic reduction that can be called on arbitrary
        NumPy arrays, as documented in :class:`pykeops.numpy.Genred`.

    Example:
        Log-likelihood of a Gaussian Mixture Model,

        .. math::
            a_i~=~f(x_i)~&=~ \log \sum_{j=1}^{N} \exp(-\gamma\cdot\|x_i-y_j\|^2)\cdot b_j \\\\
               ~&=~ \log \sum_{j=1}^{N} \exp\\big(-\gamma\cdot\|x_i-y_j\|^2 \,+\, \log(b_j) \\big).

        >>> log_likelihood = generic_logsumexp(
        ...     '(-(g * SqNorm2(x - y))) + b', # Formula
        ...     'a = Vi(1)',              # Output: 1 scalar per line
        ...     'x = Vi(3)',              # 1st input: dim-3 vector per line
        ...     'y = Vj(3)',              # 2nd input: dim-3 vector per line
        ...     'g = Pm(1)',              # 3rd input: vector of size 1
        ...     'b = Vj(1)')              # 4th input: 1 scalar per line
        >>> x = np.random.randn(1000000, 3)
        >>> y = np.random.randn(2000000, 3)
        >>> g = np.array([.5])            # Parameter of our GMM
        >>> b = np.random.rand(2000000, 1)  # Positive weights...
        >>> b = b / b.sum()               # Normalized to get a probability measure
        >>> a = log_likelihood(x, y, g, np.log(b))  # a_i = log sum_j exp(-g*|x_i-y_j|^2) * b_j
        >>> print(a.shape)
        (1000000, 1)
    """
    _, cat, _, _ = get_type(output)
    return Genred(formula, list(aliases), reduction_op='LogSumExp', axis=cat2axis(cat), **kwargs)


def generic_argkmin(formula, output, *aliases, **kwargs):
    """Alias for :class:`pykeops.numpy.Genred` with an "ArgKMin" reduction.

    Args:
        formula (string): Scalar-valued symbolic KeOps expression, as in :class:`pykeops.numpy.Genred`.
        output (string): An identifier of the form ``"AL = TYPE(K)"`` 
            that specifies the category and dimension of the output variable. Here:

              - ``AL`` is a dummy alphanumerical name.
              - ``TYPE`` is a *category*. One of:

                - ``Vi``: indexation by :math:`i` along axis 0; reduction is performed along axis 1.
                - ``Vj``: indexation by :math:`j` along axis 1; reduction is performed along axis 0.

              - ``K`` is an integer, the number of values to extract.

        *aliases (strings): List of identifiers, as in :class:`pykeops.numpy.Genred`.

    Keyword Args:
        dtype (string, default = ``"float64"``): Specifies the numerical **dtype** of the input and output arrays. 
            The supported values are:

              - **dtype** = ``"float32"``,
              - **dtype** = ``"float64"``.

    Returns:
        A generic reduction that can be called on arbitrary
        NumPy arrays, as documented in :class:`pykeops.numpy.Genred`.

    Example:
        Bruteforce K-nearest neighbors search in dimension 100:

        >>> knn = generic_argkmin(
        ...     'SqDist(x, y)',   # Formula
        ...     'a = Vi(3)',      # Output: 3 scalars per line
        ...     'x = Vi(100)',    # 1st input: dim-100 vector per line
        ...     'y = Vj(100)')    # 2nd input: dim-100 vector per line
        >>> x = np.random.randn(5,     100)
        >>> y = np.random.randn(20000, 100)
        >>> a = knn(x, y)
        >>> print(a)
        [[ 9054., 11653., 11614.],
         [13466., 11903., 14180.],
         [14164.,  8809.,  3799.],
         [ 2092.,  3323., 18479.],
         [14433., 11315., 11841.]]
        >>> print( np.linalg.norm(x - y[ a[:,0].astype(int) ], axis=1) )  # Distance to the nearest neighbor
        [10.7933, 10.3235, 10.1218, 11.4919, 10.5100]
        >>> print( np.linalg.norm(x - y[ a[:,1].astype(int) ], axis=1) )  # Distance to the second neighbor
        [11.3702, 10.6550, 10.7646, 11.5676, 11.1356]
        >>> print( np.linalg.norm(x - y[ a[:,2].astype(int) ], axis=1) )  # Distance to the third neighbor
        [11.3820, 10.6725, 10.8510, 11.6071, 11.1968]
    """
    _, cat, k, _ = get_type(output)
    return Genred(formula, list(aliases), reduction_op='ArgKMin', axis=cat2axis(cat), opt_arg=k, **kwargs)


def generic_argmin(formula, output, *aliases, **kwargs):
    """Alias for :class:`pykeops.numpy.Genred` with an "ArgMin" reduction.

    Args:
        formula (string): Scalar-valued symbolic KeOps expression, as in :class:`pykeops.numpy.Genred`.
        output (string): An identifier of the form ``"AL = TYPE(1)"`` 
            that specifies the category and dimension of the output variable. Here:

              - ``AL`` is a dummy alphanumerical name.
              - ``TYPE`` is a *category*. One of:

                - ``Vi``: indexation by :math:`i` along axis 0; reduction is performed along axis 1.
                - ``Vj``: indexation by :math:`j` along axis 1; reduction is performed along axis 0.

        *aliases (strings): List of identifiers, as in :class:`pykeops.numpy.Genred`.

    Keyword Args:
        dtype (string, default = ``"float64"``): Specifies the numerical **dtype** of the input and output arrays. 
            The supported values are:

              - **dtype** = ``"float32"``,
              - **dtype** = ``"float64"``.

    Returns:
        A generic reduction that can be called on arbitrary
        NumPy arrays, as documented in :class:`pykeops.numpy.Genred`.

    Example:
        Bruteforce nearest neighbor search in dimension 100:

        >>> nearest_neighbor = generic_argmin(
        ...     'SqDist(x, y)',   # Formula
        ...     'a = Vi(1)',      # Output: 1 scalar per line
        ...     'x = Vi(100)',    # 1st input: dim-100 vector per line
        ...     'y = Vj(100)')    # 2nd input: dim-100 vector per line
        >>> x = np.random.randn(5,     100)
        >>> y = np.random.randn(20000, 100)
        >>> a = nearest_neighbor(x, y)
        >>> print(a)
        [[ 8761.],
         [ 2836.],
         [  906.],
         [16130.],
         [ 3158.]]
        >>> dists = np.linalg.norm(x - y[ a.view(-1).long() ], axis=1)  # Distance to the nearest neighbor
        >>> print(dists)
        [10.5926, 10.9132,  9.9694, 10.1396, 10.1955]
    """
    _, cat, _, _ = get_type(output)
    return Genred(formula, list(aliases), reduction_op='ArgMin', axis=cat2axis(cat), **kwargs)

