## pre-build script to update DESCRIPTION file
# only tested on GNU/Linux

# project and package directories
proj_dir <- rprojroot::find_root(".git/index")
pkg_dir <- file.path(proj_dir, "rkeops")

# prebuild function
prebuild <- function() {
    ## DESCRIPTION file
    filename <- file.path(pkg_dir, "DESCRIPTION")
    # update Version
    command <- paste0("sed -i -e ",
                      "\"s/Version: .*/Version: $(cat ", 
                      file.path(projdir, "rkeops", "version"), ")/\" ",
                      filename)
    tmp <- system(command)
    # update Date
    command <- paste0("sed -i -e ",
                      "\"s/Date: .*/Date: ", Sys.Date(), "/\" ",
                      filename)
    tmp <- system(command)
}

# run
prebuild()
