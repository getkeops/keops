#!/bin/bash

# setup
source setup.sh

# prepare R requirements
Rscript ${CIDIR}/prepare_ci.R

# build website
Rscript ${CIDIR}/build_website.R
