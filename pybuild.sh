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
    echo "   -v     Verbose mode"
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
PYTEST_VERBOSE=0

# Get the options
while getopts 'hv' option; do
    case $option in
        h) # display Help
            print_help
            ;;
        v) # enable verbosity
            PYTEST_VERBOSE=1
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
PROJDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

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
# Installing keopscore (using pyproject.toml)                                 #
################################################################################

logging "-- Installing keopscore (editable install via pyproject.toml)..."

# With no setup.py present, pip will use pyproject.toml to build keopscore
pip install -e "${PROJDIR}/keopscore"

################################################################################
# Installing pykeops (using pyproject.toml)                                    #
################################################################################

logging "-- Installing pykeops (editable install with test extras via pyproject.toml)..."

# The [test] extras (if defined in pyproject.toml) will be installed as well.
pip install -e "${PROJDIR}/pykeops[test]"

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
