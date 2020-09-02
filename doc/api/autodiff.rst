Autodiff engine
###############

KeOps provides a simple **Automatic Differentiation** (AD) engine for generic formulas.
This feature can be used **seamlessly** through the ``Grad`` instruction
or the **PyTorch backend**: users don't have to understand backpropagation
to enjoy our "free" gradients.
Nevertheless, for the sake of completeness, here is
a short introduction to the inner workings of KeOps.

Backprop 101
============

Gradient of a vector valued function
------------------------------------

Let :math:`F:\mathbb R^n \to \mathbb R^d` be a smooth function.
Around any given point :math:`x \in \mathbb R^n`, its variations are encoded in a linear application :math:`\text{d}F(x):\mathbb R^n \to \mathbb R^d` called the **differential**
of :math:`F`: for all variation :math:`\delta x \in \mathbb R^n` of :math:`x`,

.. math::
   F(x+\delta x)
   = F(x)+ [\text{d}F(x)](\delta x) + o(\delta x).

Going further, we can define the **gradient** of :math:`F` at :math:`x`
as the adjoint of :math:`\text{d}F(x)`: it is the unique application
:math:`\partial F(x)=\text{d}F(x)^*:\mathbb R^d \to \mathbb R^n` such that
for all vector :math:`e \in \mathbb R^d` and
variation :math:`\delta x \in \mathbb R^n` of :math:`x`, we have:

.. math::
   \langle [\partial F(x)](e) , \delta x \rangle_{\mathbb R^n}
   = \langle e , [\text{d}F(x)](\delta x) \rangle_{\mathbb R^d},

where :math:`\langle\,\cdot\,,\,\cdot\,\rangle` denotes the standard scalar product.

When :math:`F` is scalar-valued (i.e. when :math:`d=1`), 
the gradient is a linear application
from :math:`\mathbb{R}` to :math:`\mathbb{R}^n`:
it is best understood as the vector of **partial derivatives** of :math:`F`.
In the general case, the matrix of the gradient in the canonical basis
is given by the **transpose of the Jacobian** of :math:`F`.




Reverse mode AD = backpropagation = chain rule
----------------------------------------------

Now, let's assume that the function :math:`F:\mathbb R^n \to \mathbb R^d` can be written as a composition :math:`F =F_p \circ \cdots \circ F_2 \circ F_1` of :math:`p` functions :math:`F_i:E_{i-1} \to E_{i}`, where :math:`E_i=\mathbb{R}^{d_i}`. With these notations :math:`d_0 = n`  and :math:`d_p = d`.  

Evaluating the gradient of :math:`F` with the `backpropagation algorithm <https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation>`_ requires:

1. A **Forward pass** to evaluate the functions

   .. math::

        \begin{array}{ccccl}
             F_i & : & E_{i-1}    & \to & E_{i} \\
             &      & x & \mapsto & F_i(x)
        \end{array}    

   and thus compute the **value** :math:`F(x)` : 

   .. figure:: ../_static/forward.svg
      :width: 80% 
      :alt: Reverse AD

2. A **Backward pass** to evaluate the (adjoints of the) differentials

   .. math::
        \begin{array}{ccccl}
	            \partial F_i & : & E_{i-1}\times E_{i} & \to & E_{i-1} \\
	             & & (x_{i-1},x_i^*) & \mapsto & [\text{d} F_i^*(x_{i-1})](x_i^*) = x_{i-1}^*
         \end{array}
    
   and compute the **gradient** of :math:`F` at location :math:`x`, applied to an arbitrary
   vector :math:`e` is the space of outputs: 

   .. figure:: ../_static/backward.svg
       :width: 80% 
       :alt: Reverse AD

This method relies on the chain-rule, as

.. math::
   \begin{align*}
    & &\text{d}(F_p\circ\cdots\circ F_1)(x_0) &= \text{d}F_p(x_{p-1}) \circ\cdots \circ \text{d} F_1(x_0),\\
    &\text{i.e.}& \text{d}(F_p\circ\cdots\circ F_1)^*(x_0) &=  \text{d} F_1^*(x_0) \circ\cdots \circ \text{d}F_p^*(x_{p-1}),\\
    &\text{i.e.}& \big[\partial F(x_0)\big](e) &= \big[\partial F_1(x_0)\big]\big( \cdots \big[\partial F_p(x_{p-1})\big](e) \big).
   \end{align*}


When :math:`F` is scalar-valued (i.e. :math:`d=1`),
this algorithm allows us to compute the vector of partial derivatives

.. math::
    \nabla F(x_0)= \big[\partial F(x_0)\big](1)

with a mere **forward-backward pass** through the computational graph of :math:`F`...
which is much **cheaper** than the naive evaluation of :math:`n` finite differences of :math:`F`.

The KeOps generic engine
========================

Backpropagation has become the standard way of computing the gradients of
arbitrary "Loss" functions in imaging and machine learning.
Crucially, any backprop engine should be able to:

- **Link together** the *forward* operations :math:`F_i` with their *backward* counterparts :math:`\partial F_i`. 
- **Store in memory** the intermediate results :math:`x_0,\dots,x_p` before using them in the backward pass.


The ``Grad`` operator
---------------------

At a low level, KeOps allows us to perform these tasks with the ``Grad`` instruction:
given a formula :math:`F`, the symbolic expression ``Grad(F, V, E)``
denotes the gradient :math:`[\partial_V F(x)] (E)` with respect to the variable :math:`V` evaluated on the input variable :math:`E`.

If ``V`` is a variable place-holder that appears in the expression of ``F``
and if ``E`` has the same dimension and category as ``F``, ``Grad(F,V,E)`` can be fed to KeOps just like any other symbolic expression. 
The resulting output will have the same dimension and category as the variable ``V``,
and can be used directly for gradient descent or **higher-order differentiation**:
operations such as ``Grad(Grad(..,..,..),..,..)`` are fully supported.


User interfaces
---------------

As evidenced by this :doc:`example <../_auto_examples/numpy/plot_generic_syntax_numpy>`, the simple ``Grad`` syntax can relieve us from the burden of differentiating symbolic formulas by hand.

Going further, our Python interface is fully compatible with the `PyTorch <https://pytorch.org/>`_ library:
feel free to use the output of a :mod:`pykeops.torch` routine **just like any other differentiable tensor**!
Thanks to the flexibility of the :mod:`torch.autograd` engine,
end-to-end automatic differentiation is at hand: 
see this :doc:`example <../_auto_examples/pytorch/plot_generic_syntax_pytorch>` or this :doc:`example <../_auto_examples/pytorch/plot_generic_syntax_pytorch_LSE>` for an introduction.


.. _`part.example2`:

An example
==========

Coming back to our :ref:`previous example <formula.example>` where the formula

.. math::

  F(p,x,y,a)_i = \left(\sum_{j=1}^N (p -a_j )^2 \exp(x_i^u + y_j^u) \right)_{i=1,\cdots,M, u=1,2,3} \in \mathbb R^{M\times 3}

.. code-block:: cpp

    F = "Sum_Reduction(Square(Pm(0,1) - Vj(3,1)) * Exp(Vi(1,3) + Vj(2,3)), 1)"

was discussed, the symbolic expression

.. code-block:: cpp

    [grad_a F] = "Grad( Sum_Reduction(Square(Pm(0,1) - Vj(3,1)) * Exp(Vi(1,3) + Vj(2,3)), 1),
                     Vj(3,1), Vi(4,3) )"


allows us to compute the gradient of :math:`F` with respect to :math:`(a_j) \in \mathbb R^N` (``= Vj(3,1)``), applied to an arbitrary test vector :math:`e\in\mathbb R^{M\times 3}` given as a fifth input ``Vi(4,3)`` :

.. math::

      \left[ [\partial_{a} F(p,x,y,a)] (e)\right]_j = - \sum_{i=1}^M \sum_{u=1}^3 2(p -a_j ) \exp(x_i^u + y_j^u) e^u_i \in \mathbb R.

With aliases, this computation simply reads:

.. code-block:: cpp

    p=Pm(0,1), x=Vi(1,3), y=Vj(2,3), a=Vj(3,1), e=Vi(4,3)
    [grad_a F](e) = "Grad( Sum_Reduction(Square(p-a)*Exp(x+y), 1), a, e)"
