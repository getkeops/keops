#!/bin/bash

set -e

# project root dir
RDIR=$(git rev-parse --show-toplevel)

# rkeops ci dir
CIDIR=${RDIR}/rkeops/ci

# setup and clean Makevars for compilation
function setup_Makevars() {
    # set up ~/.R/Makevars file for compilation
    # !! over-write ${HOME}/.R/Makevars !!
    if [[ ! -d ${HOME}/.R ]]; then mkdir -p ${HOME}/.R; fi
    if [[ -f ${HOME}/.R/Makevars ]]; then
        cp ${HOME}/.R/Makevars ${HOME}/.R/Makevars.bak.rkeops_compile
        echo "!! local ~/.R/Makevars backed up to ~/.R/Makevars.bak.rkeops_compile"
    fi
    cat ${CIDIR}/Makevars > ${HOME}/.R/Makevars
}

function cleanup() {
    if [[ -f ${HOME}/.R/Makevars.bak.rkeops_compile ]]; then
        cp ${HOME}/.R/Makevars.bak.rkeops_compile  ${HOME}/.R/Makevars
        rm ${HOME}/.R/Makevars.bak.rkeops_compile
        echo "!! local ~/.R/Makevars restored from ~/.R/Makevars.bak.rkeops_compile"
    fi
}

# set up ~/.R/Makevars file for compilation
setup_Makevars
trap cleanup EXIT

# set up .Rprofile files
export R_PROFILE_USER=${CIDIR}/.Rprofile

# prepare R requirements
Rscript ${CIDIR}/prepare_ci.R

# check package build
Rscript ${CIDIR}/run_check.R

# test package
Rscript ${CIDIR}/run_tests.R
