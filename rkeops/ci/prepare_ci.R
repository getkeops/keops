# dev package
pkg_list <- c("devtools", "roxygen2", "remotes", "rprojroot")
install_pkg(pkg_list) # function defined in local .Rprofile

# package requirements
pkg_list <- setdiff(
    remotes::local_package_deps(pkg_dir, dependencies = TRUE),
    c("stats", "utils")
)
install_pkg(pkg_list) # function defined in local .Rprofile
