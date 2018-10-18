Autodiff engine
===============

KeOps has an internal automatic differentiation engine for symbolic formulas -- compatible with the PyTorch autograd package -- that allows us to "bootstrap" all the derivatives required by the user (including gradients of gradients, etc.).
Feel free to use the output of ``Genred`` as any other torch tensor!

The ``Grad`` operator
---------------------





The ``Grad_WithSavedForward`` operator
--------------------------------------
