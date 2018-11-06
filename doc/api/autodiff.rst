Autodiff engine
===============

KeOps has an internal Automatic Differentiation (AD) engine for symbolic formulas. 

Derivative of vector valued function
------------------------------------

Let :math:`n,d\in\mathbb N` and assume that we need to compute the variations of a function :math:`F:\mathbb R^n \to \mathbb R^d` at some point :math:`x \in \mathbb R^n`. These variations are encoded in a linear application :math:`dF(x):\mathbb R^n \to \mathbb R^d` called the differential. When :math:`F` is scalar valued (ie :math:`d=1`) the (adjoint of the) differential is precisely the gradient.

When :math:`x\mapsto F(x)` is not scalar valued, its "gradient" should be understood as the adjoint of the differential operator :math:`dF^*(x): \mathbb R^d \to \mathbb R^n`, i.e. as the linear operator that takes as input a new variable :math:`e \in \mathbb R^d`  and outputs a variable :math:`g = [dF^*(x)](e) \in \mathbb R^n` such that for all variation :math:`\delta x \in \mathbb R^n` of :math:`x \in \mathbb R^n` we have:

 .. math::

    \langle [dF(x)](\delta x), e \rangle_{\mathbb R^d}  =  \langle \delta x, g \rangle_{\mathbb R^n}  =  \langle \delta x , [dF^*(x)](e) \rangle_{\mathbb R^n}



Reverse mode AD == backpropagating == chain rules
-------------------------------------------------

In what follows, :math:`E_i` denotes a finite dimensional real vector space (i.e :math:`\mathbb R^{d_i}` with :math:`d_i\in \mathbb N`). 

Assume now that the function :math:`F:\mathbb R^n \to \mathbb R^d` is written as a composition :math:`F =F_p \circ \cdots \circ F_2 \circ F_1` of :math:`p` functions :math:`F_i:E_{i-1} \to E_{i}`. Note that, in this case, :math:`d_0 = n`  and :math:`d_p = d`.  

Backpropagating through a computational graph to compute the value of the gradient requires:

1. A **Forward pass** to evaluate the functions

   .. math::

        \begin{array}{ccccl}
             F_i & : & E_{i-1}    & \to & E_{i} \\
             &      & x & \mapsto & F_i(x)
        \end{array}    

   to compute the actual value of :math:`F`: 

   .. figure:: ../_static/AD_forward.svg
      :width: 100% 
      :alt: Reverse AD

2. A **Backward pass** to evaluate the (adjoints of the) differentials

   .. math::
        \begin{array}{ccccl}
	            \partial_x F_i^* & : & E_{i-1}\times E_{i} & \to & E_{i-1} \\
	             & & (x_0,a) & \mapsto & [\partial_x F_i^*(x_0)](a)
         \end{array}
    
   to compute the value of the (adjopint of the)  gradient of :math:`F`: 

   .. figure:: ../_static/AD_backward.svg
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

