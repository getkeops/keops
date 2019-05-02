"""
=========================================
K-NN on the MNIST dataset - PyTorch API
=========================================

The :func:`pykeops.torch.generic_argkmin` routine allows us
to perform **bruteforce k-nearest neighbors search** with four lines of code.
It can thus be used to implement a **large-scale** 
`K-NN classifier <https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm>`_,
**without memory overflows** on the 
full `MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset.
"""

#####################################################################
# Setup 
# -----------------
# Standard imports:

import time
from matplotlib import pyplot as plt

import numpy as np
import torch

from pykeops.torch import generic_argkmin

use_cuda = torch.cuda.is_available()
tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

######################################################################
# Load the MNIST dataset: 70,000 images of shape (28,28).

try:
    from sklearn.datasets import fetch_openml
except ImportError:
    raise ImportError("This tutorial requires Scikit Learn version >= 0.20.")

from sklearn.model_selection import train_test_split

mnist = fetch_openml('mnist_784', cache=False)

x = tensor( mnist.data.astype('float32') )
y = tensor( mnist.target.astype('int64') )

######################################################################
# Split it into a train and test set:

D = x.shape[1]
Ntrain, Ntest = (60000, 10000) if use_cuda else (1000, 100)
x_train, y_train = x[:Ntrain,:], y[:Ntrain]
x_test,  y_test  = x[Ntrain:Ntrain+Ntest,:], y[Ntrain:Ntrain+Ntest]


######################################################################
# K-Nearest Neighbors search
# ----------------------------
# Perform the K-NN classification on 10,000 test images in dimension 784:
#

K = 3  # N.B.: K has very little impact on the running time

# Define our KeOps kernel:
knn_search = generic_argkmin( 
    'SqDist(x,y)',  # A simplistic squared L2 distance
    'ind = Vi({})'.format(K),  # The K output indices are indexed by "i"
    'x = Vi({})'.format(D),    # 1st arg: target points of dimension D, indexed by "i"
    'y = Vj({})'.format(D))    # 2nd arg: source points of dimension D, indexed by "j"

start = time.time()    # Benchmark:
ind_knn = knn_search(x_test, x_train)  # Samples <-> Dataset, (N_test, K)
lab_knn = y_train[ind_knn]  # (N_test, K) of integers in [0,9]
y_knn, _ = lab_knn.mode()   # Compute the most likely label
if use_cuda: torch.cuda.synchronize()
end = time.time()

error = (y_knn != y_test).float().mean().item()
time  = end - start

print("{}-NN on the full MNIST dataset: test error = {:.2f}% in {:.2f}s.".format(K, error*100, time))

######################################################################
# Fancy display: looks good!

plt.figure(figsize=(12,8))
for i in range(6):
    ax = plt.subplot(2,3,i+1)
    ax.imshow( (255 - x_test[i]).view(28,28).detach().cpu().numpy(), cmap="gray" )
    ax.set_title( "label = {}".format(y_knn[i].int()) )
    plt.axis('off')

plt.show()
