.. raw:: html

    <style type="text/css">
    .thumbnail {{
        position: relative;
        float: left;
        margin: 10px;
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

KErnel OPerationS, with autodiff, without memory overflows
=================================================================

.. raw:: html

    <div style="clear: both"></div>
    <div class="container-fluid hidden-xs hidden-sm">
      <div class="row">
        <a href="examples/scatterplot_matrix.html">
          <div class="col-md-2 thumbnail">
            <img src="_static/scatterplot_matrix_thumb.png">
          </div>
        </a>
        <a href="examples/errorband_lineplots.html">
          <div class="col-md-2 thumbnail">
            <img src="_static/errorband_lineplots_thumb.png">
          </div>
        </a>
        <a href="examples/different_scatter_variables.html">
          <div class="col-md-2 thumbnail">
            <img src="_static/different_scatter_variables_thumb.png">
          </div>
        </a>
        <a href="examples/horizontal_boxplot.html">
          <div class="col-md-2 thumbnail">
            <img src="_static/horizontal_boxplot_thumb.png">
          </div>
        </a>
        <a href="examples/regression_marginals.html">
          <div class="col-md-2 thumbnail">
            <img src="_static/regression_marginals_thumb.png">
          </div>
        </a>
        <a href="examples/many_facets.html">
          <div class="col-md-2 thumbnail">
            <img src="_static/many_facets_thumb.png">
          </div>
        </a>
      </div>
    </div>
    <br>

   <div class="container-fluid">
     <div class="row">
       <div class="col-md-6">

KeOps is a `cpp/cuda library <./cpp/generic-syntax>`_ that comes with bindings in `python <./python/Installation>`_ (numpy and pytorch), `Matlab <./matlab/Installation>`_ or R (coming soon). KeOps computes efficiently **Kernel dot products**, **their derivatives** and **other similar operations** on the GPU. It provides good performances and linear (instead of quadratic) memory footprint through a minimal interface.


.. raw:: html

       </div>
       <div class="col-md-6">
         <div class="panel panel-default">   
           <div class="panel-heading">
             <h3 class="panel-title">Pages</h3>
           </div>
       <div class="panel-body">

.. toctree::
   :titlesonly:

   home
   api/generic-syntax
   api/math-operations
   cpp/generic-syntax
   python/index
   python/installation
   python/generic-syntax
   matlab/index
   matlab/installation
   matlab/generic-syntax
   auto_examples/index
   road-map

.. raw:: html

       </div>
     </div>
   </div>
   
   </div>
   </div>
