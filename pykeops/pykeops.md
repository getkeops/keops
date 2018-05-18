```
                        oooo    oooo             .oooooo.                                88       
                        `888   .8P'             d8P'  `Y8b                             .8'`8.     
 oo.ooooo.  oooo    ooo  888  d8'     .ooooo.  888      888 oo.ooooo.   .oooo.o       .8'  `8.    
  888' `88b  `88.  .8'   88888[      d88' `88b 888      888  888' `88b d88(  "8      .8'    `8.   
  888   888   `88..8'    888`88b.    888ooo888 888      888  888   888 `"Y88b.      .8'      `8.  
  888   888    `888'     888  `88b.  888    .o `88b    d88'  888   888 o.  )88b    .8'        `8. 
  888bod8P'     .8'     o888o  o888o `Y8bod8P'  `Y8bood8P'   888bod8P' 8""888P'    88oooooooooo88 
  888       .o..P'                                           888                  
 o888o      `Y8P'                                           o888o                 
```


# Requirements

- a unix-like system (typically Linux or MacOs X)
- Cmake>=2.9
- A C++ compiler (gcc>=4.8, clang or nvcc)
- Python **3** with packages : numpy, GPUtil (installed via pip)
- optional : Cuda (>=9.0 is recommended), PyTorch>=0.4

# Installation

## Using pip (recommended)

Warning : A good starting point is to check your python and pip path. 
In a terminal carefully verify the consistency of the output of the following commands ```which python```,
```python --version```, ```which pip``` and ```pip --version```.

- check that cmake is working on your system by running in a terminal 
```cmake --version```. If not, run
```bash
pip install cmake
``` 
to get a proper version of cmake.
- finally, 
```bash
pip install pykeops
```

## From source

Warning: We assume here that  cmake is working.

- Clone the git repo. 
- Manually add the directory ```/path/to/keopslab/``` to you python path. 
A simple way to do it is 
```python
import os.path
import sys
sys.path.append('/path/to/keopslab/')
```
changing /path/to to the right path.

## Troubleshooting

First of all, make sure that you are using a recent C/C++ compiler (say, gcc/g++-4.8 or above of clang);
otherwise, CUDA compilation may fail in unexpected ways.
On Linux, this can be done simply by using [update-alternatives](https://askubuntu.com/questions/26498/choose-gcc-and-g-version).

Note that you can activate a "verbose" compilation mode by adding these lines *after* your KeOps imports:

```python
import pykeops
pykeops.common.compile_routines.verbose = True
```

Then, if you installed from source and recently updated KeOps, make sure that your
`keops/build/` folder (the cache of already-compiled formulas) has been emptied.

# Getting started with pyKeOps

1) Install pykeops package.

2) Run the out-of-the-box working examples located in [`./pykeops/examples/`](./pykeops/examples/) and [`./pykeops/tutorials/`](./pykeops/tutorials/).

If you are already familiar with the LDDMM theory and want to get started quickly, please check the shapes toolboxes: [plmlab.math.cnrs.fr/jeanfeydy/shapes_toolbox](https://plmlab.math.cnrs.fr/jeanfeydy/shapes_toolbox) and [plmlab.math.cnrs.fr/jeanfeydy/lddmm_pytorch](https://plmlab.math.cnrs.fr/jeanfeydy/lddmm_pytorch).


