## pre-build script to update DESCRIPTION file
# only tested on GNU/Linux

# project and package directories
projdir <- system("git rev-parse --show-toplevel", intern = TRUE)
pkgdir <- file.path(projdir, "rkeops")

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
    # remove previous Collate field and listed files
    command <- paste0("sed -i '/^Collate:/,$ d' ", filename)
    tmp <- system(command)
    # add new Collate field at end of file with currant file list
    files <- paste(c("Collate:", 
                     list.files(path="rkeops/R", recursive = TRUE)), 
                   collapse = "\n    ")
    command <- paste0("printf '%b' \"", files, "\n\" >> ", filename)
    tmp <- system(command)
}
