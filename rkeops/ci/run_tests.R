project_root_dir <- system("git rev-parse --show-toplevel", intern=TRUE)
devtools::test(file.path(project_root_dir, "rkeops"), reporter = 'fail') #, stop_on_failure=TRUE)