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


## Additional notes

### Exemple of compilation of Cpp code exported to R

Focus on required include and lib

* `RcppExports.cpp`
```
g++ -std=gnu++11 -I"/usr/share/R/include" -DNDEBUG  -I"~/R/x86_64-pc-linux-gnu-library/3.6/Rcpp/include" -I"~/R/x86_64-pc-linux-gnu-library/3.6/RcppEigen/in
clude"   -fpic  -g -O2 -fdebug-prefix-map=/build/r-base-p1m1mT/r-base-3.6.1=. -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 -g  
-c RcppExports.cpp -o RcppExports.o
```

* `rkeops.cpp`
```
g++ -std=gnu++11 -I"/usr/share/R/include" -DNDEBUG  -I"~/R/x86_64-pc-linux-gnu-library/3.6/Rcpp/include" -I"~/R/x86_64-pc-linux-gnu-library/3.6/RcppEigen/include"   -fpic  -g -O2 -fdebug-prefix-map=/build/r-base-p1m1mT/r-base-3.6.1=. -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 -g  -c rkeops.cpp -o rkeops.o
```

* shared lib
```
g++ -std=gnu++11 -shared -L/usr/lib/R/lib -Wl,-Bsymbolic-functions -Wl,-z,relro -o rkeops.so RcppExports.o rkeops.o -L/usr/lib/R/lib -lR
```

* required include and lib can be found with
```R
R.home("include")
system.file("include", package = "Rcpp")
system.file("include", package = "RcppEigen")
R.home("lib")
```