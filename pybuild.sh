#!/usr/bin/env bash

################################################################################
#  Instructions for Creating a New Release
#
#  0) Generate a Twine API token and configure your `.pypirc` file
#     Ensure both TestPyPI and PyPI are properly set up.
#
#  1) Update the version number in the file: ./keops_version and Changes log: ./CHANGELOG.md
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
#     Note: TestPyPI may have dependency resolution issues.
#     If problems occur, install pykeops from PyPI, uninstall it,
#     then reinstall pykeops from TestPyPI.
#
#     Note: Do not forget to remove the install from TestPyPI.
#
#  5) Once validated, upload to the official PyPI:
#       twine upload ./build/dist/keopscore-XXXXX.tar.gz
#       twine upload ./build/dist/pykeops-XXXXX.tar.gz
################################################################################

set -euo pipefail

readonly PYTHON_BIN="python3"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
readonly BUILD_VENV="${SCRIPT_DIR}/.build_venv"
readonly BUILD_REQUIREMENTS=(pip build pyclean)
readonly VERSION="$(<"${SCRIPT_DIR}/keops_version")"

LOCAL_PYBUILD=0
PYBUILD_VERBOSE=0

print_help() {
    cat <<EOF
Build script for keopscore/pykeops packages.

Usage: $0 [option...]

   -h     Print the help
   -l     Build in local mode (without hard-coded keopscore version requirement in pykeops)
   -v     Verbose mode

Note: by default, the keopscore version requirement is hard-coded in pykeops.
EOF
}

log_verbose() {
    if [[ "${PYBUILD_VERBOSE}" -eq 1 ]]; then
        printf '%b\n' "$1"
    fi
}

parse_options() {
    while getopts ":hlv" option; do
        case "${option}" in
            h)
                print_help
                exit 0
                ;;
            l)
                LOCAL_PYBUILD=1
                log_verbose "## local build (keopscore version requirement is not hard-coded in pykeops)"
                ;;
            v)
                PYBUILD_VERBOSE=1
                log_verbose "## verbose mode"
                ;;
            \?)
                echo "Error: Invalid option"
                exit 1
                ;;
        esac
    done
}

prepare_setup() {
    local setup_file="${SCRIPT_DIR}/pykeops/setup.py"
    local backup_file="${setup_file}.pybuild.bak"

    if [[ "${LOCAL_PYBUILD}" -eq 1 ]]; then
        return
    fi

    log_verbose "-- Preparing setup for version ${VERSION}..."
    cp "${setup_file}" "${backup_file}"
    sed -i -e 's/"keopscore"/"keopscore==" + current_version/' "${setup_file}"
}

cleanup_setup() {
    local setup_file="${SCRIPT_DIR}/pykeops/setup.py"
    local backup_file="${setup_file}.pybuild.bak"

    if [[ ! -f "${backup_file}" ]]; then
        return
    fi

    log_verbose "-- Restoring pykeops setup..."
    mv "${backup_file}" "${setup_file}"
}

prepare_python_environment() {
    log_verbose "-- Preparing python environment for build..."
    "${PYTHON_BIN}" -m venv --clear "${BUILD_VENV}"

    # shellcheck disable=SC1091
    source "${BUILD_VENV}/bin/activate"

    log_verbose "---- Python version = $(${PYTHON_BIN} -V)"
    "${PYTHON_BIN}" -m pip install -U "${BUILD_REQUIREMENTS[@]}"
}

clean_python_sources() {
    log_verbose "-- Cleaning Python sources before build..."
    pyclean "${SCRIPT_DIR}/keopscore"
    pyclean "${SCRIPT_DIR}/pykeops"
}

build_source_distribution() {
    local package_name="$1"
    local package_path="$2"

    log_verbose "-- Building ${package_name}..."
    "${PYTHON_BIN}" -m build --sdist --outdir "${SCRIPT_DIR}/build/dist" "${package_path}"
}

main() {
    parse_options "$@"
    prepare_setup
    trap cleanup_setup EXIT
    prepare_python_environment
    clean_python_sources
    build_source_distribution "keopscore" "${SCRIPT_DIR}/keopscore"
    build_source_distribution "pykeops" "${SCRIPT_DIR}/pykeops"
}

main "$@"
