project_root_dir <- system("git rev-parse --show-toplevel", intern=TRUE)
devtools::install(file.path(project_root_dir, "rkeops"))
