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
    # R file list
    filename_list <- list.files(path=file.path(pkgdir, "R"), recursive = TRUE)
    full_filename_list <- list.files(path=file.path(pkgdir, "R"), 
                                     recursive = TRUE, full.names = TRUE)
    R_files <- which(grepl(pattern=".R$", filename_list))
    filename_list <- filename_list[R_files]
    full_filename_list <- full_filename_list[R_files]
    # check for empty files (or files with only commented lines)
    commands <- paste0("grep -vc '^#' ", full_filename_list, " || true")
    tmp <- as.numeric(unlist(lapply(commands, system, intern=TRUE)))
    nonempty_files <- which(tmp>0)
    # add new Collate field at end of file with current file list
    files <- paste(c("Collate:", filename_list[nonempty_files]), 
                   collapse = "\n    ")
    command <- paste0("printf '%b' \"", files, "\n\" >> ", filename)
    tmp <- system(command)
}
