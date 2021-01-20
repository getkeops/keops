# RKeOps Continuous Integration (CI)

## Automatic test

For check the [requirements](#requirements) section below to configure your system.

To run the CI of RKeOps 
```bash
bash run_ci.sh
```

> **Note 1**: required R packages and dependencies for the CI are installed in the directory `${HOME}/.R_libs_keops_ci`.

> **Note 2**: this script creates a file `${HOME}/.R/Makevars` to add options to R compilation engine (it backs up any pre-existing version of the file).

To enable automatic tests to be run on GPU, the environment variable `TEST_GPU` 
should be defined with the value `1` before running the CI scripts, i.e.
```bash
export TEST_GPU=1
```

## Details

### Requirements

Before running any R script, you should run the environment setup script (to configure a local `.Renviron` file) with:
```bash
bash setup_Renviron.sh
```

To install the requirements, you must run:
```bash
Rscript prepare_ci.R
```

### Troubleshooting

`devtools` package dependencies 
  - `apt-get install build-essential libcurl4-gnutls-dev libxml2-dev libssl-dev` (ubuntu)

Configuration failed to find libgit2 library. Try installing:
  - brew: libgit2 (MacOS)
  - deb: libgit2-dev (Debian, Ubuntu, etc)
  - rpm: libgit2-devel (Fedora, CentOS, RHEL)

### Tests and checks

To run the package automatic tests, you must run:
```bash
Rscript run_tests.R
```

To run the R package check (equivalent to `R CMD build` and `R CMD check`), you must run:
```bash
Rscript run_tests.R
```

#Vignette to online doc conversion

To generate html files from the vignettes that can be integrated into the 
Sphinx doc, see this script [`./html2doc.sh`](./html2doc.sh).

> Note: on MacOS, you need to install pdflatex to run check, e.g. with `brew cask install basictex`.

