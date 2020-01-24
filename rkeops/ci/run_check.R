project_root_dir <- system("git rev-parse --show-toplevel", intern=TRUE)
devtools::check(file.path(project_root_dir, "rkeops"), error_on = "error")