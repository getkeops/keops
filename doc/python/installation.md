# Python Install

PyKeOps contains the python 3 bindings for the cpp/cuda library KeOps. It provides 
standard python functions that can be used in any python 3 code.

## Requirements

- A unix-like system: typically Linux or MacOs X 
- A C++ compiler: gcc>=5, clang
- Cmake>=3.10
- Ninja 
- **Python 3** with packages: numpy, GPUtil (installed via pip or conda)
- Cuda (Optional): version >=9.0 is recommended but it works with older version. Warning: check the compatibility of C++ compiler
- PyTorch (Optional): version >=0.4.1


## Using pip (recommended)

1. Just in case: in a terminal, verify the **consistency** of the outputs of the commands `which python`, `python --version`, `which pip` and `pip --version`. You can then install the dependencies with:

    ```bash
    pip install numpy
    pip install GPUtil
    pip install cmake
    pip install ninja
    ```

2. In a terminal

    ```bash
    pip install pykeops
    ```

    Note that the compiled shared objects (*.so) will be stored into the folder  ```~/.cache/libkeops-$version``` where ```~``` is the path to your home folder and ```$version``` is the package version number.

3. Test your installation: [as explained below](#test)

## From source using git

1. Clone keops library repo at a location of your choice (here denoted as ```/path/to```)

    ```bash
    git clone https://plmlab.math.cnrs.fr/benjamin.charlier/libkeops.git /path/to/libkeops
    ```

      Note that your compiled .so routines will be stored into the folder `/path/to/libkeops/build` : this directory must have **write permission**.

2. Manually add the directory `/path/to/libkeops` (and **not** `/path/to/libkeops/pykeops/`) to your python path.
    + This can be done once and for all, by adding the path to to your `~/.bashrc`. In a terminal,
        ```bash
        echo "export PYTHONPATH=$PYTHONPATH:/path/to/libkeops/" >> ~/.bashrc

        ``` 
    + Otherwise, you can add the following line to the beginning of your python scripts:
    
        ```python
        import os.path
        import sys
        sys.path.append('/path/to/libkeops')
        ```

3. Test your installation: [as explained below](#test)



## Testing your installation <A name='test'></A>

1. In a python terminal,
    ```python
    import numpy as np
    import pykeops.numpy as pknp

    x = np.arange(1,4).reshape(1,-1)
    y = np.arange(3,6).reshape(1,-1)
    b = np.arange(3,0,-1).reshape(1,-1)
    p = np.array([0.25]).reshape(1,-1)

    my_conv = pknp.Genred("Exp(-p*SqNorm2(x-y))*b", ["p = Pm(1)", "x = Vx(3)", "y = Vy(3)", "b = Vy(3)"])
    print(my_conv(p.astype('float32'),x.astype('float32'),y.astype('float32'),b.astype('float32')))
    ```

    should return

    ```python
    [[0.1493612  0.09957413 0.04978707]]
    ```

2. If you use pytorch, the following code:

    ```python
    import torch
    import pykeops.torch as pktorch

    x = torch.arange(1,4, dtype=torch.float32).view(1,-1)
    y = torch.arange(3,6, dtype=torch.float32).view(1,-1)
    b = torch.arange(3,0,-1, dtype=torch.float32).view(1,-1)
    p = torch.tensor([.25], dtype=torch.float32)

    my_conv = pktorch.Genred("Exp(-p*SqNorm2(x-y))*b", ["p = Pm(1)", "x = Vx(3)", "y = Vy(3)", "b = Vy(3)"])
    print(my_conv(p,x,y,b))
    ```

    should return

    ```python
    tensor([[ 0.1494,  0.0996,  0.0498]])
    ```

## Troubleshooting

### Compilation issues

First of all, make sure that you are using a C++ compiler which is compatible with your nvcc (CUDA) compiler may fail in unexpected ways. On Debian based Linux distros, this can be done simply by using [update-alternatives](https://askubuntu.com/questions/26498/choose-gcc-and-g-version).

### Verbosity level

Note that you can activate a "verbose" compilation mode by adding these lines *after* your KeOps imports:

```python
import pykeops
pykeops.verbose = True
```

### Clean build cache <div id=cache_dir></div>

If you experience problems with compilation (or numerical inaccuracies after a KeOps update), it may be a good idea to flush the build folder (the cache of already-compiled formulas). To get the directory name:

```python
print(pykeops.build_folder)
```
