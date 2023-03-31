# Organize TODOs/tasks/actions related to RKeOps package

## Implement RKeOps v2+

See [PR #279](https://github.com/getkeops/keops/pull/279)

Switch from the old keops engine based on C++/Cmake to the new `keopscore` framework based on a meta-programming engine.

Implementation : directly use `PyKeOps` through `reticulate` to avoid maintaining two different binders for Python and R

TODOs
- [ ] proof-of-concept using `PyKeOps` from R through `reticulate`
- [ ] integrate `PyKeOps` management inside `RKeOps` with `reticualte`
- [ ] reimplement `keops_kernel()` and `keops_grad()` functions
- [ ] integrate LazyTensor ([PR #192](https://github.com/getkeops/keops/pull/192))
- [ ] clean RKeOps from deprecated and oldies (unused code from copyright files, `.Rbuildignore` file)
