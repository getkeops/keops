# requirements
if(! "pkgdown" %in% .packages(all.available = TRUE)) install.packages("pkgdown")

# package config
proj_dir <- rprojroot::find_root(".git/index")
pkg_dir <- file.path(proj_dir, "rkeops")

# reticulate config
envname <- "rkeops-ci"
if(reticulate::virtualenv_exists(envname))
    reticulate::virtualenv_remove(envname, confirm = FALSE)
reticulate::virtualenv_create(envname)
reticulate::virtualenv_install(envname, "pykeops")
reticulate::use_virtualenv(virtualenv = envname, required = TRUE)
reticulate::py_config()

# load package
devtools::load_all(pkg_dir)

# build website
pkgdown::build_site(pkg = pkg_dir)
