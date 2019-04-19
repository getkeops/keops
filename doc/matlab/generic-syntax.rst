Matlab API
==========

The example described below is implemented in the example Matlab script `script_GenericSyntax.m <https://github.com/getkeops/keops/blob/master/keopslab/examples/script_GenericSyntax.m>`_ located in directory ``keopslab/examples``.

The Matlab bindings provide a function `keops_kernel <https://github.com/getkeops/keops/blob/master/keopslab/generic/keops_kernel.m>_` which can be used to define the corresponding convolution operations. Following the previous example, one may write

.. code-block:: matlab
     
     f = keops_kernel('Square(p-a)*Exp(x+y)','p=Pm(1)','a=Vj(1)','x=Vi(3)','y=Vj(3)');

which defines a Matlab function handler ``f`` which can be used to perform a sum reduction for this formula:

.. code-block:: matlab
    
    c = f(p,a,x,y);


where ``p``, ``a``, ``x``, ``y`` must be arrays with compatible dimensions as previously explained. A gradient function `keops_grad <https://github.com/getkeops/keops/blob/master/keopslab/generic/keops_grad.m>_` is also provided. For example, to get the gradient with respect to ``y`` of the previously defined function ``f``, one needs to write:

.. code-block:: matlab
    
    Gfy = keops_grad(f, 'y');

which returns a new function that can be used as follows :

.. code-block:: matlab

    Gfy(p, a, x, y, e)

where ``e`` is the input gradient array (here of type ``Vi(3)``).
