## pre-build script to update DESCRIPTION file
# only tested on GNU/Linux

# project and package directories
projdir <- system("git rev-parse --show-toplevel", intern = TRUE)
pkgdir <- file.path(projdir, "rkeops")

# clean function (before building)
clean <- function() {
    ## clean inst/build
    build_dir <- file.path(pkgdir, "inst", "build")
    if(dir.exists(build_dir)) {
        cmd <- paste0("rm -rf ", build_dir, "/*")
        tmp <- system(cmd)
    }
}

# prebuild function
prebuild <- function() {
    ## DESCRIPTION file
    filename <- file.path(pkgdir, "DESCRIPTION")
    # update Version
    command <- paste0("sed -i -e ",
                      "\"s/Version: .*/Version: $(cat ", 
                      file.path(projdir, "version"), ")/\" ",
                      filename)
    tmp <- system(command)
    # update Date
    command <- paste0("sed -i -e ",
                      "\"s/Date: .*/Date: ", Sys.Date(), "/\" ",
                      filename)
    tmp <- system(command)
}

# run
clean()
prebuild()
