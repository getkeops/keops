This library is under development... some keys features have not been implemented yet. 

```
__/\\\\\\____     _______     __/\\\________     _______________     _______________        
 _\////\\\____     _______     _\/\\\________     __/\\\_________     _______________       
  ____\/\\\____     __/\\\_     _\/\\\________     _\/\\\_________     ___/\\\\\\\\\__      
   ____\/\\\____     _\///__     _\/\\\________     _\/\\\\\\\\____     __/\\\/////\\\_     
    ____\/\\\____     __/\\\_     _\/\\\\\\\\\__     _\/\\\////\\\__     _\/\\\\\\\\\\__    
     ____\/\\\____     _\/\\\_     _\/\\\////\\\_     _\/\\\\\\\\/___     _\/\\\//////___   
      ____\/\\\____     _\/\\\_     _\/\\\__\/\\\_     _\/\\\///\\\___     _\/\\\_________  
       __/\\\\\\\\\_     _\/\\\_     _\/\\\\\\\\\__     _\/\\\_\///\\\_     _\/\\\_________ 
        _\/////////__     _\///__     _\/////////___     _\///____\///__     _\///__________
```


libkp is a library which implements in Cuda various operations using kernels. We also provide bindings in python (numpy and pytorch complient),  matlab and R.

It uses a "tiled implementation" in order to have a $`O(n)`$ memory footprint instead of usual $`O(n^2)`$ codes generating by high level libraries like Thrust or cuda version of pyTorch and TensorFlow. It comes with various examples ranging from lddmm (non rigid deformations) to kernel density estimations (non parametric statistics).  

For instance, the basic example is a Gaussian convolution on a non regular grid in $`\mathbb R^3`$ : given two point clouds $`(x_i)_{i=1}^N \in  \mathbb R^{N \times 3}`$ and $`(y_j)_{j=1}^M \in  \mathbb R^{M \times 3}`$  and a vector field $`(\beta_j)_{j=1}^M \in  \mathbb R^{M \times 3}`$ attached to the $`y_j`$'s, libkp may computes $`(\gamma_i)_{i=1}^N \in  \mathbb R^{N \times 3}`$ given by
```math
 \gamma_i =  \sum_j K(x_i,y_j) \beta_j,  \qquad i=1,cdots,N
```
 where $`K(x_i,y_j) = \exp(-|x_i - y_j|^2 / \sigma^2)`$. The best performances are achieved for the range $`N=1000`$ to $`N=5.10^5`$.
 
# Quick start

## Python user

Two steps:

1) Compilation of the cuda codes. The subdirectory `./python` contains a shell script `makefile.sh`. The user needs to custom the paths contained in this file. The python wrappers use the ctypes library to produce python function callable from any python script. 

2) Run the out-of-the-box working examples `./python/example/convolution.py`

3) If you are already familiar with the LDDMM theory and want ot get started quickly, please check the all-in-one script: `python/tutorials/lddmm/lddmm_pytorch.py`

## Matlab user

Two steps:

1) Compilation of the cuda codes. The subdirectory `./matlab` contains a shell script `makefile.sh`. The user needs to custom the paths contained in this file. The script produces mex files callable from any matlab script.

2) Run the out-of-the-box working examples `./matlab/example/convolution.m`

## R user

To do.


   
authors : Charlier,Feydy, Glaunès
