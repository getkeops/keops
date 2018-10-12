.. raw:: html

    <style type="text/css">
    .thumbnail {{
        position: relative;
        float: left;
        margin: 30px;
        width: 180px;
        height: 200px;
    }}

    .thumbnail img {{
        position: absolute;
        display: inline;
        left: 0;
        width: 170px;
        height: 170px;
    }}

    </style>

.. figure:: _static/logo/keops_logo.png
   :height: 200px
   :alt: Keops logo

Presentation
------------

KeOps is a library that computes on a GPU **generic reductions** of 2d arrays whose entries may be computed through a mathematical formula. We provide an autodiff engine to generate effortlessly the formula of the derivative. For instance, KeOps can compute **Kernel dot products**, **their derivatives**.
::
    from pykeops import Genred

    # compile the function computing the derivative of a gaussian convolution
    my_conv = Genred(formula='Grad(Exp(SqNorm2(x-y)/2), x, b)',
                     alias=['x = Vx(3)', 'y = Vy(3)', 'b = Vx(3)']
                     red_type='Sum')
    
    # ... aply it to some data. Get the result in an array
    result = my_conv(x,y,b)

It provides good performances and linear (instead of quadratic) memory footprint. It handles multi GPU. More details are provided in :doc:`here <api/why_using_keops>`.

Installation
------------

The core of KeOps relies on a set of C++/CUDA routines for which we provide bindings in the following languages:

* :doc:`Python (numpy or pytorch) <python/installation>`
* :doc:`Matlab <matlab/installation>`
* :doc:`C++ API <cpp/generic-syntax>`

Tutorial and examples
---------------------

.. raw:: html

    <div style="clear: both"></div>
    <div class="container-fluid hidden-xs hidden-sm">
      <div class="row">
        <a href="auto_examples/index.html">
          <div class="col-md-2 thumbnail">
            <img src="_static/thumbs/gaussian_mixture.png">
          </div>
        </a>
        <a href="auto_examples/kmeans.html">
          <div class="col-md-2 thumbnail">
            <img src="_static/thumbs/kmeans.png">
          </div>
        </a>
        <a href="auto_examples/index.html">
          <div class="col-md-2 thumbnail">
            <img src="_static/thumbs/LDDMM_surface.png">
          </div>
        </a>
        <a href="auto_examples/index.html">
          <div class="col-md-2 thumbnail">
            <img src="_static/thumbs/optimal_transport.png">
          </div>
        </a>
        <a href="auto_examples/index.html">
          <div class="col-md-2 thumbnail">
            <img src="_static/thumbs/wasserstein_150.png">
          </div>
        </a>
      </div>
    </div>
    <br>

   <div class="container-fluid">
     <div class="row">
       <div class="col-md-6">


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

.. raw:: html

       </div>
       <div class="col-md-6">
         <div class="panel panel-default">   
           <div class="panel-heading">
             <h3 class="panel-title">Pages</h3>
           </div>
       <div class="panel-body">

.. toctree::
   :maxdepth: 2
   :caption: Presentation

   api/why_using_keops
   api/road-map
   api/math-operations
   api/generic-syntax2
   auto_examples/index

.. toctree::
   :maxdepth: 2
   :caption: PyKeops

   python/index
   python/installation
   python/generic-syntax
   python/kernel-product

.. toctree::
   :maxdepth: 2
   :caption: KeopsLab

   matlab/index
   matlab/installation
   matlab/generic-syntax

.. toctree::
   :maxdepth: 2
   :caption: Keops

   cpp/generic-syntax


.. raw:: html

       </div>
     </div>
   </div>
   
   </div>
   </div>
