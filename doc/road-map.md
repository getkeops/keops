# Road map

In future releases, we will focus on **performances**
and implement heuristics / search algorithms for optimized compilation parameters.

A **reference paper** will soon be put on Arxiv.

- [ ] R bindings
- [ ] Check the limits of the GPU device, gridsizes
- [ ] Rules for choosing 1D/2D: this should be made at the c++ level !
- [ ] Speed benchmarks; Symbolic Diff vs. Hardcoded vs. PyTorch
- [ ] Get rid of ctypes?
- [ ] Sums of kernels:
  - [ ] `Map<F,V>` operation
  - [ ] `Sum<d,V>` operation
  - [ ] `kernel_product` support

Warning : ctype is not compatible with cpp templated function as it does not support name mangling (ie it need a fixed name for each function). Template are made by copy/paste code. See :  https://stackoverflow.com/questions/9084111/is-wrapping-c-library-with-ctypes-a-bad-idea


