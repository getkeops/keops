#!/bin/bash

# setup
source setup.sh

# prepare R requirements
Rscript ${CIDIR}/prepare_ci.R

# check package build
Rscript ${CIDIR}/run_check.R

# test package
Rscript ${CIDIR}/run_tests.R
