.. _`part.engine_formulas`:

========================
Generic formulas
========================

The two previous sections have higlighted the need for **efficient
Map-Reduce GPU routines** in data sciences. To complete our guided tour of
the inner workings of the KeOps library, we now explain how
**generic reductions and formulas** are encoded within our C++
codebase.


Note that this section is extracted
from the Chapter 2.2 of Jean Feydy's PhD thesis,
to be available soon.


.. toctree::
   :maxdepth: 2
   :caption: Generic formulas

   design_choices.rst
   math_operations.rst
   reductions.rst
   backpropagation.rst