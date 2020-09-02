# Comparisons with other frameworks

This set of scripts can be used to benchmark the performance of KeOps versus other solutions.

## Results

We compare the performances of several computing libraries on a simple benchmark: a Gaussian kernel matrix-vector product with a growing number of points `N` in dimension `D = 3`.  All experiments are performed with `float32` precision on a Nvidia RTX 2080 TiGPU, with the exception of the PyTorch-TPU column that was run in Google Colab.


|              |   PyTorch    |   PyTorch-TPU   |    TF-XLA      |   Halide        |        TVM       |     PyKeOps   |    KeOps++   |
|:------------:|:------------:|:---------------:|:--------------:|:---------------:|:----------------:|:-------------:|:------------:|
|N = 10k       |     9 ms     |      10 ms      |     13 ms      |  1.0 ms         |     3.8 ms       | **0.7 ms**    |  **0.40 ms** |
|N = 100k      | out of mem   |   out of mem    |      89 ms     |   34.1 ms       |     36.8 ms      | **15.0 ms**   |  **14.6 ms** |
|N = 1M        | out of mem   |   out of mem    |    out of mem  |   3.8 s         |     2.79 s       | **1.39 s**    |  **1.38 s**  |
|Lines of code |     5        |     5           |       5        |    15           |      17          |     5         |    55        |
|Interface     |  NumPy-like  |     NumPy-like  |   NumPy-like   |        C++      | low-level Python | NumPy-like    |   C++        |

## Scripts

### PyKeOps

To compute the Gaussian convolution on GPU using Pykeops with torch.

1. intall pykeops

```bash
$ pip install pykeops
```

2. Run

```bash
$ python KeOps.py
```

### KeOps++

To compute the Gaussian convolution on GPU using KeOps c++ backend

```bash
mkdir build
cd build
cmake ../../keops
make test_fromdevice
example/test_fromdevice 100000
```

### [Halide](https://halide-lang.org/)

To compute the Gaussian convolution on GPU using Halide (c++ code)

1. Download Halide, and extract it in the current folder E.g.

```bash
$ wget https://github.com/halide/Halide/releases/download/release_2019_08_27/halide-linux-64-gcc53-800-65c26cba6a3eca2d08a0bccf113ca28746012cc3.tgz
$ tar xvf halide-linux-64-gcc53-800-65c26cba6a3eca2d08a0bccf113ca28746012cc3.tgz
```
2. Compile the sources launch it:

```bash
$ g++ gauss_conv_halide.cpp -g -std=c++11 -I halide/tutorial -I halide/include -I halide/tools -L halide/bin -lHalide -lpthread -ldl -o gauss_conv_halide
$ LD_LIBRARY_PATH=halide/bin ./gauss_conv_halide 100000
```

### [TF-XLA](https://www.tensorflow.org/xla)

To compute the Gaussian convolution on GPU using TF-XLA:

1. Install TensorFlow-XLA 2.xx with gpu support. On Google Colab,

```python
!pip install tensorflow-gpu==2.0.0
```
should do the trick. On a linux system, you may need to define the XLA-FLAGS env: 

```bash
$ export XLA_FLAGS="--xla_gpu_cuda_data_dir=/path/to/cuda
```

2. Run

```bash
$ python TF_XLA.py
```

### [Pytorch GPU](https://pytorch.org/)

To compute the Gaussian convolution on GPU using PyTorch:

```bash
$ python Pytorch_GPU.py
```

### Pytorch TPU

To compute the Gaussian convolution on TPU using PyTorch. This code should be run on a Google Colab session, with TPU acceleration. Copy/paste the content of `PyTorch_TPU.py` in the collab notebook.

### [TVM](https://tvm.apache.org/)

To compute the Gaussian convolution on GPU using PyTorch:

1. Install TVM using this [tutorial](https://docs.tvm.ai/install/index.html) 
2. If you are running on a linux system, export `TVM_HOME` and `PYTHONPATH` variable.
3. Run
```bash
$ python TVM.py
```
