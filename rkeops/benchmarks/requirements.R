# dev helper packages
pkg_list <- c("devtools", "roxygen2", "remotes", "rprojroot")
install.packages(pkg_list)

proj_dir <- rprojroot::find_root(".git/index")
pkg_dir <- file.path(proj_dir, "rkeops")

# package requirements
pkg_list <- setdiff(
    remotes::local_package_deps(pkg_dir, dependencies = TRUE),
    c("stats", "utils")
)
install.packages(pkg_list)