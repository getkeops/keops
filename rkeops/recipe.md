# Recipe to build rkeops

In this file are described all steps required to build the `rkeops` package. 
You can either use Rstudio machinery with the [attached](#rstudio) Rstudio 
project file, or use R commands and follow the recipe described 
[below](#r-command-tools)


## Rstudio

You can use Rstudio for development. To set up, the package project,
you can use the attached [project file](../keops.Rproj).

You will be able to document, build and check `rkeops` with 
Rstudio tools (`document`, `build`, `check`).

We recommend to use the function `prebuild()` available with
`source(rkeops/prebuild.R)` to update the [DESCRIPTION](.rkeops/DESCRIPTION) 
file of `rkeops` (date, version, included R files) before 
building.


## R command tools

Keops root directory
```R
projdir <- system("git rev-parse --show-toplevel", intern = TRUE)
pkgdir <- file.path(projdir, "rkeops")
```

### Preliminary
(done once at begining, not required)

* R package skeleton
```R
setwd(projdir)
package.skeleton("rkeops")
```

* Enable `testthat`, `Rcpp`, `RcppEigen`
```R
setwd(pkgdir)
usethis::use_testthat()
usethis::use_rcpp()
usethis::use_rcpp_eigen()
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
prebuild()
```
**Note:** The `Collate` field with included R files should be the last one in 
the [DESCRIPTION](.rkeops/DESCRIPTION) file for this script to work.

### Package build and check

```R
setwd(pkgdir)
Rcpp::compileAttributes()
devtools::build()
devtools::check()
```



### Release

```R
devtools::release()
```