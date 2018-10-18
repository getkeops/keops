
# Documentation for KeOps

This folder contains source files used to build the KeOps documentation webpage. As a regular user of KeOps, you do not need to use files in this folder, you can get KeOps documentation through the following link:
[KeOps online website](https://www.kernel-operations.io)

If you want to rebuild KeOps documentation, here are the steps: 

* first install required packages via pip:

```
pip install sphinx sphinx-gallery recommonmark sphinxcontrib-httpdomain sphinx_rtd_theme
```
* Then do

```
make html
```
Note that this will run Python examples and tutorials contained in pykeops/examples and pykeops/tutorials, so you should first make sure they run ok on your system.
