Road map
========

In future releases, we will focus on **performances**
and implement heuristics / search algorithms for optimized compilation parameters.


- Put **reference paper** on Arxiv.
- R bindings.
- Check the limits of the GPU device, gridsizes.
- Rules for choosing 1D/2D: this should be made at the c++ level !
- Speed benchmarks; Symbolic Diff vs. Hardcoded vs. PyTorch.
- Allow users to inject their own CUDA subroutines for maximum efficiency.
- Sums of kernels:
  - `Map<F,V>` operation.
  - `Sum<d,V>` operation.
  - `kernel_product` support.

