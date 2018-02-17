- [ ] R bindings
- [ ] Python tutorials
- [x] generic cuda codes
- [x] baseline CPU implementation
- [x] compile without Cuda
- [x] support pointer in device global mem
- [x] implement non-scalar kernels (Glaunès & Michelli)
- [x] everybody knows how to use Git !

For the 20th of Dec.:

Jean :
- [x] Adapt cudaconv.py to make it generic
- [x] PyTorch recursive syntax
- [ ] implement non-isotropic kernels (diagonal, full symmetric tensor)
- [ ] Derivatives w.r.t. \Sigma (scalar, diagonal, full) + (fixed, tied, independent)
- [ ] Let KernelProduct choose 1D/2D, CPU/GPU
- [ ] Gaussian Mixtures
- [ ] Other Applications

Benjamin :
- [x] 1D scheme for the generic code
- [ ] R bindings
- [ ] Check the limits of the GPU device, gridsizes
- [ ] Rules for choosing 1D/2D: this should be made at the c++ level ! 
- [ ] CMake shoul be specific to each binding to avoid overhead


Joan :
- [ ] Full on-device : fix bug in FromDevice 
- [ ] Speed benchmarks; Symbolic Diff vs. Hardcoded vs. PyTorch
- [x] Adapt mex files to cmake compilation

