# Organize TODOs/tasks/actions related to RKeOps package

## Merge R LazyTensor

See [PR #192](https://github.com/getkeops/keops/pull/192)

Remaining issues:

- [ ] output for
  + Min_ArgMin (missing argmin)
  + Max_ArgMax (same)
  + KMin (dim of output)
  + argKMin (dim of output)
  + KMin_ArgKMin (dim of output)
  + LogSumExp (check output value)
  + SumSoftMaxWeight (Python error: "Axis should be 0 or 1")

> More in [`dev/debug_LazyTensor.R`] and [`dev/debut_LazyTensor.py`]

- [ ] fix inline doc for min, max (following sum), and other default functions

- [ ] revamp vignettes regarding LazyTensor, and mention it in main vignette

- [ ] fix issue from `devtools::check()`

## Implement RKeOps v2+

See [PR #279](https://github.com/getkeops/keops/pull/279)

Switch from the old keops engine based on C++/Cmake to the new `keopscore` framework based on a meta-programming engine.

Implementation : directly use `PyKeOps` through `reticulate` to avoid maintaining two different binders for Python and R

TODOs
- [x] proof-of-concept using `PyKeOps` from R through `reticulate`
- [x] integrate `PyKeOps` management inside `RKeOps` with `reticualte`
- [x] reimplement `keops_kernel()` and `keops_grad()` functions
- [x] integrate LazyTensor ([PR #192](https://github.com/getkeops/keops/pull/192))
- [x] clean RKeOps from deprecated and oldies (unused code from copyright files, `.Rbuildignore` file)
