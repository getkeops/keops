The winning combination
=================================


Recent advances in “AI” have been driven by the diffusion of two critical
pieces of software:

-  **Automatic differentiation.** Relying on symbolic “historical
   records” that are attached to the program’s variables, modern computing
   libraries now implement transparent :mod:`.grad()` operators.
   |br|

-  **GPU backends for tensor-based computations.** Benefitting from the
   long-term investment of Nvidia, recent frameworks provide backends
   for *convolutions* and *linear algebra* operators that harness the
   parallel computing power of modern hardware.

Enabling the large scale tuning of the weights that parameterize
*convolutional neural networks*, these components were first paired
together in a Python package by the 
`Theano <http://deeplearning.net/software/theano/>`_ library, 
developed between 2007 and 2018 by the 
`MILA institute <https://mila.quebec/en/>`_. 
Today, using the Google and Facebook-backed 
`TensorFlow <https://www.tensorflow.org/>`_
and `PyTorch <https://pytorch.org/>`_ libraries, 
**tens of thousands of users** routinely optimize objective functions that
involve millions of parameters using gradient descent strategies.

In less than ten years, deep learning frameworks have allowed **“GPU”** and
**“backpropagation”** to become **ubiquitous buzzwords** in applied sciences.
However, outside of the 
`graphics <https://developer.nvidia.com/gpugems/GPUGems/gpugems_pref01.html>`_ and
`autodiff <http://www.autodiff.org/?module=Tools&tool=TAPENADE>`_ communities, 
few researchers make
the effort of understanding the inner workings of these convenient
black-boxes. In a world where 
**fast runtimes make or break the popularity of research fields**, 
this oversight has effectively surrendered most of
the scientific initiative in machine learning and image processing to
the lead developers of TensorFlow and PyTorch.

.. list-table::

  * - .. figure:: images/dragon_1000.jpg
         :alt: Subsampled dragon.

         ..

         **(a)** Subsampled model, 11,102 triangles.

    - .. figure:: images/dragon_full.jpg
         :alt: Full dragon.

         ..

         **(b)** Full model, 871,414 triangles.


**Figure.**
Illustrating the gap between the performances of “machine learning“ and
“graphics“ routines with subsampled copies of the 
`Stanford dragon <http://graphics.stanford.edu/data/3Dscanrep/>`_. 
**(a)** Due to intrinsic limitations of
the tensor-centric paradigm implemented by TensorFlow and
PyTorch, modern Gaussian Processes packages cannot scale to datasets
with more than 10,000 samples without making 
significant approximations
or `mobilizing high-end GPU chips for days <https://arxiv.org/abs/1903.08114>`_. 
**(b)** Relying on a tailor-made CUDA
backend, the KeOps library allows mathematicians to catch-up with
the state-of-the-art and handle large datasets (i.e. point clouds) with
a convenient interface.

**A key limitation: the narrow focus on CNNs.**
Since the days of Theano and 
its `Lasagne extension <https://lasagne.readthedocs.io/en/latest/>`_, 
deep learning frameworks have always prioritized the support of stacked
*convolution* and *fully connected* layers – to the detriment of other
algorithmic structures. Among mathematicians, this lack of investment in
*general purpose* frameworks has led to a 
**strong underrating of modern hardware**: 
let’s just cite the common belief, held in the Machine
Learning community, that the ceiling for exact kernel methods on a
single device lies around :math:`10^4` samples... 
At a time when off-the-shelf graphical
engines render `millions of triangles <https://www.youtube.com/watch?v=pNmhJx8yPLk>`_ 
at 60 Frames Per Second on gaming laptops.

**Our contribution:** stepping outside of the tensor-centric paradigm.
**Bringing graphics-like performances to our fellow mathematicians** is the
main goal of this work: in 2020, researchers should be allowed to stay
*creative* without having to compromise too much on *performances*.

After a brief, **high-level crash-course on backpropagation and the**
**intricacies of GPU programming**, we will present 
the **inner workings** of the KeOps Map-Reduce engine.
Through a convenient symbolic abstraction, the “LazyTensor” wrapper,
KeOps provides efficient routines for machine learning 
and computational geometry *without ever
compromising on usability*.



.. |br| raw:: html

  <br/><br/>

