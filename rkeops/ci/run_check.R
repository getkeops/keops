proj_dir <- rprojroot::find_root(".git/index")
pkg_dir <- file.path(proj_dir, "rkeops")
devtools::check(pkg_dir, error_on = "error")