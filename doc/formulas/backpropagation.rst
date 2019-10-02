Backpropagation
================================


Last but not least, **KeOps fully supports automatic differentiation**.
Most of the magic required is implemented by the :mod:`F::DiffT<V,GRADIN>`
attributes of KeOps formulas and reductions, as discussed in
:doc:`previous pages <math_operations>`.



Backprop through a Sum reduction
-------------------------------------------

Then, to implement the PyTorch :mod:`backward` of the KeOps :mod:`Genred`
operator, we simply have to remember that if
:math:`(g_i) \in \mathbb{R}^{\mathrm{M}\times \mathrm{E}}` is a “gradient to backpropagate” with
respect to the output :math:`(a_i) \in \mathbb{R}^{\mathrm{M}\times \mathrm{E}}` 
of a :mod:`Genred`
call with a **Sum reduction**, we can write that for all variations
:math:`(\delta p,\delta x_i, \delta y_j)` of the parameters, :math:`i`-
and :math:`j`-variables, at order 1:

.. math::

   \begin{aligned}
   \Big\langle&
   \sum_{j=1}^\mathrm{N} F(p+\delta p, x_i+\delta x_i, y_j + \delta y_j)
   - F(p, x_i, y_j)~,~
   g_i
   \Big\rangle_{\mathbb{R}^{\mathrm{M}\times E}}\\
   ~=~&
   \sum_{i=1}^\mathrm{M} \sum_{j=1}^\mathrm{N} 
   \Big(  
   \left\langle \partial_p F(\dots) \cdot g_i,  \delta p \right\rangle
   \,+\,
   \left\langle \partial_{x_i} F(\dots) \cdot g_i,  \delta x_i \right\rangle
   \,+\,
   \langle \partial_{y_j} F(\dots) \cdot g_i,  \delta y_j \rangle  
   \Big).\end{aligned}

Consequently, **performing the appropriate permutations of sums**:

.. math::

   \begin{aligned}
   \partial_{x_i}
   \Big[ \sum_{j=1}^\mathrm{N} F(p,x_i,y_j)\Big] \cdot (g_i)
   ~&=~ 
   \phantom{\sum_{i=1}^\mathrm{M} }
   \sum_{j=1}^\mathrm{N} \Big( 
   \partial_{x_i}
   \Big[ F(p,x_i,y_j)\Big] \cdot g_i \Big), \\
   \partial_{y_j}
   \Big[ \sum_{j=1}^\mathrm{N} F(p,x_i,y_j)\Big] \cdot (g_i)
   ~&=~ 
   \phantom{\sum_{i=1}^\mathrm{M} }
   \sum_{i=1}^\mathrm{M}  \Big( 
   \partial_{y_j}
   \Big[ F(p,x_i,y_j)\Big] \cdot g_i \Big), \\
   \partial_{p}
   \Big[ \sum_{j=1}^\mathrm{N} F(p,x_i,y_j)\Big] \cdot (g_i)
   ~&=~ 
   \sum_{i=1}^\mathrm{M} \sum_{j=1}^\mathrm{N} \Big( 
   \partial_{p}
   \Big[ F(p,x_i,y_j)\Big] \cdot g_i \Big).\end{aligned}


Backprop through a Log-Sum-Exp reduction
-------------------------------------------

Similarly, when :math:`(a_i)` is given through a **Log-Sum-Exp
reduction**:

.. math::

   \begin{aligned}
   a_i~=~ \log \sum_{j=1}^\mathrm{N} \exp F(p,x_i,y_j),\end{aligned}

straightforward computations show that:

.. math::

   \begin{aligned}
   \partial_{x_i}
   \Big[ \log \sum_{j=1}^\mathrm{N} \exp F(p,x_i,y_j)\Big] \cdot (g_i)
   ~&=~ 
   \phantom{\sum_{i=1}^\mathrm{M} }
   \sum_{j=1}^\mathrm{N} 
   e^{F(p,x_i,y_j) - a_i}\cdot
   \Big(
   \partial_{x_i}
   \Big[ F(p,x_i,y_j)\Big] \cdot g_i\Big), \\
   \partial_{y_j}
   \Big[ \log \sum_{j=1}^\mathrm{N} \exp F(p,x_i,y_j)\Big] \cdot (g_i)
   ~&=~ 
   \phantom{\sum_{i=1}^\mathrm{M} }
   \sum_{i=1}^\mathrm{M} 
   e^{F(p,x_i,y_j) - a_i}\cdot
   \Big(
   \partial_{y_j}
   \Big[ F(p,x_i,y_j)\Big] \cdot g_i\Big), \\
   \partial_{p}
   \Big[ \log \sum_{j=1}^\mathrm{N} \exp F(p,x_i,y_j)\Big] \cdot (g_i)
   ~&=~ 
   \sum_{i=1}^\mathrm{M} \sum_{j=1}^\mathrm{N}
   e^{F(p,x_i,y_j) - a_i}\cdot
   \Big(
   \partial_{p}
   \Big[ F(p,x_i,y_j)\Big] \cdot g_i\Big).\end{aligned}

In other words, a **backward pass** through a :mod:`Genred` call that involves
a Sum or a Log-Sum-Exp reduction can 
**always be written as a symbolic Map-Reduce computation**.

Bootstrapping derivatives of arbitrary order
--------------------------------------------------

Applying these **commutation rules** between the differential operator
:math:`\partial_\texttt{V}` and the Sum or Log-Sum-Exp reductions, the
`pykeops/torch/generic/generic_red.py <https://github.com/getkeops/keops/blob/master/pykeops/torch/generic/generic_red.py>`_ 
module provides full
compatibility between KeOps :mod:`LazyTensors` and the 
:mod:`torch.autograd`
package. Thanks to recursive calls to the :mod:`Genred` 
operator and to our
symbolic math engine, **everything works just fine** – even high-order
derivatives.