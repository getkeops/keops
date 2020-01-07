# Recipe to build rkeops

In this file are described all steps required to build the `rkeops` package. 
You can either use Rstudio machinery with the [attached](#rstudio) Rstudio 
project file, or use R commands and follow the recipe described 
[below](#r-command-tools)


## Rstudio

You can use Rstudio for development. To set up, the package project,
you can use the attached [project file](../keops.Rproj).

You will be able to document, build, test and check `rkeops` with 
Rstudio tools (`document`, `build`, `test` `check`).

We recommend to run:
```R
source(rkeops/prebuild.R)
```
to update the [DESCRIPTION](.rkeops/DESCRIPTION) 
file of `rkeops` (date, version, included R files) before 
building.


## R command tools

Keops root directory
```R
projdir <- system("git rev-parse --show-toplevel", intern = TRUE)
pkgdir <- file.path(projdir, "rkeops")
```

### R documentation

* Automatic doc generation with `roxygen2`
```R
setwd(pkgdir)
devtools::load_all()
devtools::document(roclets = c('rd', 'collate', 'namespace', 'vignette'))
```

### Prebuild script

A prebuild script is available to update the `DESCRIPTION` file of `rkeops`
(date, version, included R files):
```R
setwd(pkgdir)
source("prebuild.R")
```

### Package build and check

```R
setwd(pkgdir)
devtools::build()
devtools::check()
```

### Release

```R
devtools::release()
```

**Note:** To release on CRAN, it is recommended to use the command line tools, c.f. next section. 

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