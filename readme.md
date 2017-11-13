This library is under development... some keys features have not been implemented yet. 

```
oooo   o8o   .o8       oooo                   
`888        "888       `888                   
888  oooo   888oooo.   888  oooo  oo.ooooo.  
888  `888   d88' `88b  888 .8P'    888' `88b 
888   888   888   888  888888.     888   888 
888   888   888   888  888 `88b.   888   888 
o888o o888o  `Y8bod8P' o888o o888o  888bod8P' 
                                   888       
                                  o888o     
```


libkp is a library which implements in Cuda various operations using kernels. It 
uses a "tiled implementation" in order to have a O(n) memory footprint instead 
of usual O(n^2) codes generating by high level libraries like Thrust, pyTorch or
TensorFlow... It comes with various examples ranging from lddmm (non rigid 
deformations) to kernel density estimations (non parametric statistics).  

For instance, the basic example is a Gaussian convolution (on a non regular grid):
```math
 \gamma_i =  \sum_j K(x_i,y_j) \beta_j
```
 where $`K(x_i,y_j) = exp(-|x_i - y_j|^2 / \sigma^2)`$.
 
The core of code is written in CUDA. We also provide bindings in python (numpy and pytorch complient),  matlab and R.


# Quick start

## Python user

Two steps:

1) Compilation of the cuda codes. The subdirectory `./python` contains a shell script `makefile.sh`. The user needs to custom the paths contained in this file.

2) Run the out-of-the-box working examples `./python/example/convolution.py`

3) If you are already familiar with the LDDMM theory and want ot get started quickly, please check the all-in-one script: `python/tutorials/lddmm/lddmm_pytorch.py`

## matlab user


## R user




   
authors : Charlier,Feydy, Glaunès
