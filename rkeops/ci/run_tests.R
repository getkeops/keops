proj_dir <- rprojroot::find_root(".git/index")
pkg_dir <- file.path(proj_dir, "rkeops")
devtools::test(pkg_dir, reporter = c('Progress', 'fail'))
