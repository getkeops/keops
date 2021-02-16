#!/bin/bash

set -e

# project root dir
RDIR=$(git rev-parse --show-toplevel)

# rkeops ci dir
CIDIR=${RDIR}/rkeops/ci

# set up ~/.R/Makevars file for compilation
bash ${CIDIR}/setup_Makevars.sh

# set up .Rprofile files
export R_PROFILE_USER=${CIDIR}/.Rprofile

# prepare R requirements
Rscript ${CIDIR}/prepare_ci.R

# check package build
Rscript ${CIDIR}/run_check.R

# test package
Rscript ${CIDIR}/run_tests.R
