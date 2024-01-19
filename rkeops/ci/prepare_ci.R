# dev package
pkg_list <- c(
    "devtools", "roxygen2", "remotes", "reticulate", "rprojroot", "withr")
install_pkg(pkg_list) # function defined in local .Rprofile

# setup
proj_dir <- rprojroot::find_root(".git/index")
pkg_dir <- file.path(proj_dir, "rkeops")

# package requirements
pkg_list <- setdiff(
    remotes::local_package_deps(pkg_dir, dependencies = TRUE),
    c("stats", "utils")
)
install_pkg(pkg_list) # function defined in local .Rprofile
