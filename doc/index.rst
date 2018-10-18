.. figure:: _static/logo/keops_logo.png
   :width: 100% 
   :alt: Keops logo

KeOps library
-------------

KeOps is a library that computes on a GPU **generic reductions** of 2d arrays whose entries may be computed through a mathematical formula. We provide an autodiff engine to generate effortlessly the formula of the derivative. For instance, KeOps can compute **Kernel dot products** and **their derivatives**. 

A typical sample of (pseudo) code looks like

.. code-block:: python

    from keops import Genred
    
    # create the function computing the derivative of a Gaussian convolution
    my_conv = Genred(reduction='Sum',
                     formula='Grad(Exp(SqNorm2(x-y) / Cst(2)), x, b)',
                     alias=['x=Vx(3)', 'y=Vy(3)', 'b=Vx(3)'])
    
    # ... apply it to the 2d array x, y, b with 3 columns and a (huge) number of lines
    result = my_conv(x,y,b)

KeOps provides good performances and linear (instead of quadratic) memory footprint. It handles multi GPU. More details are provided here:

* :doc:`Documentation <api/why_using_keops>`.
* :doc:`Learning KeOps syntax with examples <_auto_examples/index>`
* :doc:`Tutorials gallery <_auto_tutorials/index>`

Installation
------------

The core of KeOps relies on a set of C++/CUDA routines for which we provide bindings in the following languages:

* :doc:`Python (numpy or pytorch) <python/installation>`
* :doc:`Matlab <matlab/installation>`
* :doc:`C++ API <cpp/generic-syntax>`

Project using KeOps
-------------------

* `Deformetrica <http://www.deformetrica.org>`_ 
* `FshapesTk <https://plmlab.math.cnrs.fr/benjamin.charlier/fshapesTk>`_
* `Shapes toolbox <https://plmlab.math.cnrs.fr/jeanfeydy/shapes_toolbox>`_

Related project
---------------

You may also be interrested in `Tensor Comprehensions <https://facebookresearch.github.io/TensorComprehensions/introduction.html>`_.

Authors
-------

Feel free to contact us for any bug report or feature request:

- `Benjamin Charlier <http://imag.umontpellier.fr/~charlier/>`_
- `Jean Feydy <http://www.math.ens.fr/~feydy/>`_
- `Joan Alexis Glaunès <http://www.mi.parisdescartes.fr/~glaunes/>`_ 

Table of content
----------------

.. toctree::
   :maxdepth: 2
   :caption: KeOps

   api/why_using_keops
   api/road-map
   api/math-operations
   api/autodiff

.. toctree::
   :maxdepth: 2
   :caption: PyKeops

   python/index
   _auto_examples/index
   _auto_tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: KeopsLab

   matlab/index

.. toctree::
   :maxdepth: 2
   :caption: Keops++

   cpp/index

