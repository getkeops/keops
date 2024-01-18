# Organize TODOs/tasks/actions related to RKeOps package

## R LazyTensor

- [x] Merge [PR #192](https://github.com/getkeops/keops/pull/192)

Remaining issues:

- [x] fix following reductions:
  + [x] Min_ArgMin (missing argmin)
  + [x] Max_ArgMax (same)
  + [x] KMin (dim of output)
  + [x] argKMin (dim of output)
  + [x] KMin_ArgKMin (dim of output)
  + [x] LogSumExp (check output value)
  + [x] SumSoftMaxWeight (Python error: "Axis should be 0 or 1")

- [x] fix inline doc for min, max (following sum), and other default functions
- [ ] fix issue with `?"/"`
- [x] fix issue with `%*%` (including code and doc)

- [x] check if concatenation is done along the inner dimension for `concat` function
- [x] fix doc of function `concat` in `R/lazytensor_operations.R`

- [x] revamp vignettes regarding LazyTensor, and mention it in main vignette

- [x] fix issue from `devtools::check()`

- [x] update website page with new vignettes

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
