Autodiff engine
===============

KeOps provides a simple Automatic Differentiation (AD) engine for generic formulas.
This feature can be used seamlessly through the ``Grad`` instruction
or the PyTorch backend: users don't have to understand backpropagation
to enjoy our "free" gradients.

Nevertheless, for the sake of completeness, here is
a short introduction to the inner workings of KeOps.

Backprop 101: Gradient of a vector valued function
-----------------------------------------------------

Let :math:`F:\mathbb R^n \to \mathbb R^d` be a smooth function of which we study the variations.
Around any given point :math:`x \in \mathbb R^n`, these are encoded in a linear application :math:`\text{d}F(x):\mathbb R^n \to \mathbb R^d` called the **differential**
of :math:`F`.

Equivalently, we can define the **gradient** of :math:`F` at :math:`x`
as the adjoint of :math:`\text{d}F(x)`: it is the unique application
:math:`\partial_x F(x)=\text{d}F(x)^*:\mathbb R^d \to \mathbb R^n` such that
for all vector :math:`e \in \mathbb R^d` and
variation :math:`\delta x \in \mathbb R^n` of :math:`x`, we have:

.. math::
   \langle ~[\partial_x F(x)](e) \,,\, \delta x ~\rangle_{\mathbb R^n}
   ~=~ \langle ~e \,,\, [\text{d}F(x)](\delta x) ~\rangle_{\mathbb R^d},

where :math:`\langle\,\cdot\,,\,\cdot\,\rangle` denotes the standard scalar product.

When :math:`F` is scalar valued (i.e. when :math:`d=1`), 
the gradient is a linear application
from :math:`\mathbb{R}` to :math:`\mathbb{R}^n`:
it is best understood as the vector of partial derivatives of :math:`F`.
In the general case, its matrix in the canonical basis
is given by the *transpose* of the Jacobian.




Reverse mode AD = backpropagation = chain rule
----------------------------------------------

Now, let's assume that the function :math:`F:\mathbb R^n \to \mathbb R^d` can be written as a composition :math:`F =F_p \circ \cdots \circ F_2 \circ F_1` of :math:`p` functions :math:`F_i:E_{i-1} \to E_{i}`, where :math:`E_i=\mathbb{R}^{d_i}`. With these notations :math:`d_0 = n`  and :math:`d_p = d`.  

Evaluating the gradient of :math:`F` with the **backpropagation algorithm** requires:

1. A **Forward pass** to evaluate the functions

   .. math::

        \begin{array}{ccccl}
             F_i & : & E_{i-1}    & \to & E_{i} \\
             &      & x & \mapsto & F_i(x)
        \end{array}    

   and thus compute the *value* :math:`F(x)` : 

   .. figure:: ../_static/forward.svg
      :width: 100% 
      :alt: Reverse AD

2. A **Backward pass** to evaluate the (adjoints of the) differentials

   .. math::
        \begin{array}{ccccl}
	            \partial_x F_i & : & E_{i-1}\times E_{i} & \to & E_{i-1} \\
	             & & (x_0,a) & \mapsto & [\text{d} F_i^*(x_0)](a)
         \end{array}
    
   and evaluate the *gradient* of :math:`F` at location :math:`x`, applied to an arbitrary
   input :math:`e` : 

   .. figure:: ../_static/backward.svg
       :width: 100% 
       :alt: Reverse AD

   The backward pass illustrates the fact the (adjoint) differential of a function composition is the composition of the (adjoint) differentials.

The ``Grad`` operator
---------------------

Given a formula ``F``, the gradient of ``F`` with respect to the variable ``V`` and taking as input the variable ``E`` may be simply computed by writing ``Grad(F, V, E)``. Let us call this new formula ``G``.

The variable ``E`` is of the same size as the output of ``F`` and the output of ``G`` is of the same size as the variable ``V``.


.. _`part.example2`:

An example 
^^^^^^^^^^

Coming back to the :ref:`example <formula.example>` where the formula 

.. code-block:: cpp

    SumReduction(Square(Pm(0,1) - Vy(3,1)) * Exp(Vx(1,3) + Vy(2,3)), 1)
    
was discussed, we want to compute the derivative of :math:`f` with respect to :math:`a \in \mathbb R^N` and applied to :math:`e\in\mathbb R^{M\times 3}`

.. math::

      \left[ [\partial_{a} f^*(p,x,y,a)] (e)\right]_j = - \sum_{i=1}^M \sum_{u=1}^3 2(p -a_j ) \exp(x_i^u + y_j^u) e^u_i \in \mathbb R

To do so, we can define a new formula corresponding to the derivative with respect to the variable ``Vy(3,1)`` with the following syntax:

.. code-block:: cpp

    Grad(SumReduction(Square(Pm(0,1) - Vy(3,1)) * Exp(Vx(1,3) + Vy(2,3)), 1), Vy(3,1), Vx(4,3))

note here that the input variable ``Vx(4,3)`` is now the fifth variable of the new formula. One can also use the syntax with aliases:

.. code-block:: cpp

    p=Pm(0,1), x=Vx(1,3), y=Vy(2,3), a=Vy(3,1), e=Vx(4,3)
    Grad(SumReduction(Square(p-a)*Exp(x+y), 1), a, e)

See also this :doc:`example <../_auto_examples/plot_generic_syntax_numpy>`.

Pytorch users
-------------

The autodiff engine is compatible with the PyTorch autograd package -- that allows us to "bootstrap" all the derivatives required by the user (including gradients of gradients, etc.).
Feel free to use the output of ``pyKeOps`` function  as any other torch tensor!

See this :doc:`example <../_auto_examples/plot_generic_syntax_pytorch>` or this :doc:`example <../_auto_examples/plot_generic_syntax_pytorch_LSE>`.

