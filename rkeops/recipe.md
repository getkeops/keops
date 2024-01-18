# Recipe to build rkeops

In this file are described all steps required to build the `rkeops` package. 
You can either use Rstudio machinery with the [attached](#rstudio) Rstudio 
project file, or use R commands and follow the recipe described 
[below](#r-command-tools).

## Set up environment

### Set up a CRAN repository

Edit or create the file `~/.Rprofile` and add:
```
## Default repo
local({r <- getOption("repos")
    r["CRAN"] <- "https://cloud.r-project.org" 
    options(repos=r)
})
```

### Install R package dependencies

1. Install devtools dependencies
```bash
# on Ubuntu
apt-get install build-essential libcurl4-gnutls-dev libxml2-dev libssl-dev
```

2. Install devtools
```R
install.packages("devtools")
```

## Development pipeline

Refer to [`dev_history.Rmd`](dev_history.Rmd) for information and details about RKeOps pipeline development.

## Continuous Integration (CI)

See the [dedicated file](./ci/README.md) for more details.

Check this [script](./ci/run_ci.sh).

To enable automatic tests to be run on GPU, the environment variable `TEST_GPU` 
should be defined with the value `1` before running the CI scripts, i.e.
```bash
export TEST_GPU=1
```

## Documentation and webpage generation

The RKeOps package webpage is generated with [`pkgdown`](https://pkgdown.r-lib.org/).

You should be able to generate it with:

```bash
cd rkeops/ci
bash build_website.sh
cd ../..
cp -r rkeops/docs/ WEBSITE_ROOT_DIR/rkeops/
```

### Legacy doc (version <2)

To generate html files from the vignettes that can be integrated into the 
Sphinx doc, see this [script](./ci/html2doc.sh).

> Note: on MacOS, you need to install pdflatex to run check, e.g. with `brew cask install basictex`.


## Rstudio

You can use Rstudio for development. To set up, the package project,
you can use the attached [project file](../keops.Rproj).

You will be able to document, build, test and check `rkeops` with 
Rstudio tools (`document`, `build`, `test` `check`).

## R command tools

Keops root directory
```R
proj_dir <- rprojroot::find_root(".git/index")
pkg_dir <- file.path(proj_dir, "rkeops")
```

### R documentation

* Automatic doc generation with `roxygen2`
```R
devtools::load_all(pkg_dir)
devtools::document(pkg_dir, roclets = c('rd', 'collate', 'namespace', 'vignette'))
```

### Package build and check

```R
devtools::build(pkg_dir)
devtools::check(pkg_dir)
```

### Release

```R
devtools::release(pkg_dir)
```

**Note:** To release on CRAN, it is recommended to use the command line tools, c.f. next section.

**Note:** Before releasing on CRAN, you can use <https://win-builder.r-project.org/> to test if the package can be built on Windows (it may not works, it just is just required to install it on Windows).

## Command line tools

To release on CRAN, you can generate the tar.gz file and check it with the following bash commands:
```bash
R CMD build rkeops
R CMD check --as-cran rkeops_<version>.tar.gz
```
(replace `<version>` by the current version number).

For submission on CRAN, visit <https://cran.r-project.org/submit.html>.


## Additional notes

### Vignette creation

You can compile vignettes (in `rkeops/vignettes`) directly in Rstudio or 
in R command line with the package knitr.
