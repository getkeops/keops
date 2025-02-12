#!/bin/bash

# exit in case of any errors
set -e

################################################################################
# help                                                                         #
################################################################################
function print_help() {
    # Display Help
    echo "Test script for keopscore/pykeops packages."
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
    if [[ ${PYTEST_VERBOSE} == 1 ]]; then
        echo -e "$1"
    fi
}

################################################################################
# process script options                                                       #
################################################################################

# default options
LOCAL_PYBUILD=0
PYTEST_VERBOSE=0

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
# script setup                                                                 #    # already checked until here (same as checkhealth)
################################################################################

# project root directory
# PROJDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # Here is the issue with CI cuda_test
PROJDIR=$(git rev-parse --show-toplevel)

# python exec
PYTHON="python3"

# python environment for test
TEST_VENV=${PROJDIR}/.test_venv

# python test requirements (names of packages to be installed with pip)
TEST_REQ="pip"

################################################################################
# prepare python environment                                                   #
################################################################################

logging "-- Preparing python environment for test..."

${PYTHON} -m venv --clear ${TEST_VENV}
source ${TEST_VENV}/bin/activate

logging "---- Python version = $(python -V)"

pip install -U ${TEST_REQ}

################################################################################
# Installing keopscore (via pyproject.toml)                                   #
################################################################################

logging "-- Installing keopscore (editable install via pyproject.toml)..."
pip install -e "${PROJDIR}/keopscore"

################################################################################
# Installing pykeops (via pyproject.toml)                                      #
################################################################################

logging "-- Installing pykeops (editable install with test extras via pyproject.toml)..."
pip install -e "${PROJDIR}/pykeops[test]"

################################################################################
# Ensure repository root is in PYTHONPATH to locate sibling packages          #
################################################################################

export PYTHONPATH="${PROJDIR}:${PYTHONPATH}"
logging "---- PYTHONPATH set to: ${PYTHONPATH}"

################################################################################
# Running keopscore tests                                                     #
################################################################################

logging "-- Running keopscore tests..."
pytest -v keopscore/keopscore/test/

################################################################################
# Running pykeops tests                                                        #
################################################################################

logging "-- Running pykeops tests..."
pytest -v pykeops/pykeops/test/