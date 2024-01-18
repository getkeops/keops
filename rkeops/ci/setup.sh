#!/bin/bash

set -e

# project root dir
RDIR=$(git rev-parse --show-toplevel)

# rkeops ci dir
CIDIR=${RDIR}/rkeops/ci

# set up .Rprofile files
export R_PROFILE_USER=${CIDIR}/.Rprofile

# setup local install dir for R libs
RLIBDIR=${CIDIR}/.R_libs
if [[ ! -d $RLIBDIR ]]; then mkdir -p $RLIBDIR; fi
export R_LIBS_USER=$RLIBDIR