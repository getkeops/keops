#!/bin/bash

RDIR=$(git rev-parse --show-toplevel)
CIDIR=${RDIR}/rkeops/ci

# set up ~/.R/Makevars file for compilation
# !! over-write ${HOME}/.R/Makevars !!
if [[ ! -d ${HOME}/.R ]]; then mkdir -p ${HOME}/.R; fi
if [[ -f ${HOME}/.R/Makevars ]]; then
    cp ${HOME}/.R/Makevars ${HOME}/.R/Makevars.bak
fi
cat ${CIDIR}/Makevars > ${HOME}/.R/Makevars