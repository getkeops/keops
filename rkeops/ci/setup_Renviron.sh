#!/bin/bash

RDIR=$(git rev-parse --show-toplevel)
CIDIR=${RDIR}/rkeops/ci
RLIB=${HOME}/.R_libs_keops_ci

if [[ ! -d ${RLIB} ]]; then mkdir ${RLIB}; fi

echo -e "R_LIBS=${RLIB}\nR_LIBS_USER=${RLIB}" > ${CIDIR}/.Renviron