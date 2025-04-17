#!/bin/bash

################################################################################
#  Instructions for Creating a New Release
#
#  0) Generate a Twine API token and configure your `.pypirc` file
#     Ensure both TestPyPI and PyPI are properly set up.
#
#  1) Update the version number in the file: ./keops_version
#
#  2) Build the packages using the build script:
#       sh ./pybuild.sh
#
#  3) Test the installation locally:
#       pip install ./build/dist/keopscore-XXXXX.tar.gz
#       pip install ./build/dist/pykeops-XXXXX.tar.gz
#
#  4) Upload to TestPyPI and validate the installation (e.g., on Colab):
#       twine upload ./build/dist/keopscore-XXXXX.tar.gz --repository testpypi
#       twine upload ./build/dist/pykeops-XXXXX.tar.gz --repository testpypi
#       pip install -i https://test.pypi.org/simple/ pykeops
#
#     ⚠️  Note: TestPyPI may have dependency resolution issues.
#        If problems occur, install pykeops from PyPI, uninstall it,
#        then reinstall pykeops from TestPyPI.
#
#     ⚠️  Note: Do not forget to remove the install from TestPyPI...
#
#  5) Once validated, upload to the official PyPI:
#       twine upload ./build/dist/keopscore-XXXXX.tar.gz
#       twine upload ./build/dist/pykeops-XXXXX.tar.gz
################################################################################



# exit in case of any errors
set -e

################################################################################
# help                                                                         #
################################################################################
function print_help() {
    # Display Help
    echo "Build script for keopscore/pykeops packages."
    echo
    echo "Usage: $0 [option...]"
    echo
    echo "   -h     Print the help"
    echo "   -l     Build in local mode (without hard-coded keopscore version requirement in pykeops)"
    echo "   -v     Verbose mode"
    echo
    echo "Note: by default, the keopscore version requirement is hard-coded in pykeops."
    echo
    exit 1
}

################################################################################
# utils                                                                        #
################################################################################

# log with verbosity management
function logging() {
    if [[ ${PYBUILD_VERBOSE} == 1 ]]; then
        echo -e $1
    fi
}

################################################################################
# process script options                                                       #
################################################################################

# default options
LOCAL_PYBUILD=0
PYBUILD_VERBOSE=0

# Get the options
while getopts 'hlv' option; do
    case $option in
        h) # display Help
            print_help
            ;;
        l) # local build (no hard-coded keopscore version requirements)
            LOCAL_PYBUILD=1
            logging "## local build (keopscore version requirements is NOT hard-coded in pykeops)"
            ;;
        v) # enable verbosity
            PYBUILD_VERBOSE=1
            logging "## verbose mode"
            ;;
        \?) # Invalid option
            echo "Error: Invalid option"
            exit 1
            ;;
    esac
done

################################################################################
# script setup                                                                 #
################################################################################

# project root directory
PROJDIR=$(git rev-parse --show-toplevel)

# python exec
PYTHON="python3"

# python environment for build
BUILD_VENV=${PROJDIR}/.build_venv

# python build requirements (names of packages to be installed with pip)
BUILD_REQ="pip build pyclean"

# KeOps current version
VERSION=$(cat ./keops_version)

################################################################################
# prepare build (and cleanup after)                                            #
################################################################################

# prepare setup and clean up on exit
function prepare_setup() {
    logging "-- Preparing setup..."
    # hard-code keopscore requirements
    if [[ ${LOCAL_PYBUILD} == 0 ]]; then
        cp ${PROJDIR}/pykeops/setup.py ${PROJDIR}/pykeops/setup.py.pybuild.bak
        sed -i -e "s/\"keopscore\"/\"keopscore==\" + current_version/" ${PROJDIR}/pykeops/setup.py
    fi
}

function cleanup_setup() {
    logging "-- Cleaning up setup..."
    cp ${PROJDIR}/pykeops/setup.py.pybuild.bak ${PROJDIR}/pykeops/setup.py
    rm ${PROJDIR}/pykeops/setup.py.pybuild.bak
}

prepare_setup
trap cleanup_setup EXIT

################################################################################
# prepare python environment                                                   #
################################################################################

logging "-- Preparing python environment for build..."

${PYTHON} -m venv --clear ${BUILD_VENV}
source ${BUILD_VENV}/bin/activate

logging "---- Python version = $(python -V)"

pip install -U ${BUILD_REQ}

################################################################################
# clean before build                                                           #
################################################################################

logging "-- Cleaning Python sources before build..."

# remove __pycache__ *.pyc
pyclean ${PROJDIR}/keopscore
pyclean ${PROJDIR}/pykeops

################################################################################
# build keopscore                                                              #
################################################################################

logging "-- Building keopscore..."

python -m build --sdist --outdir ${PROJDIR}/build/dist ${PROJDIR}/keopscore

################################################################################
# build pykeops                                                                #
################################################################################

logging "-- Building pykeops..."

python -m build --sdist --outdir ${PROJDIR}/build/dist ${PROJDIR}/pykeops
