#!/usr/bin/env bash

set -euo pipefail

readonly PYTHON_BIN="python3"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
readonly TEST_VENV="${SCRIPT_DIR}/.test_venv_pytest"
readonly TEST_REQUIREMENTS=(pip)

PYTEST_VERBOSE=0

print_help() {
    cat <<EOF
Test script for keopscore/pykeops packages.

Usage: $0 [option...]

   -h     Print the help
   -v     Verbose mode
EOF
    exit 1
}

log_verbose() {
    if [[ "${PYTEST_VERBOSE}" -eq 1 ]]; then
        printf '%b\n' "$1"
    fi
}

parse_options() {
    while getopts ":hv" option; do
        case "${option}" in
            h)
                print_help
                ;;
            v)
                PYTEST_VERBOSE=1
                log_verbose "## verbose mode"
                ;;
            \?)
                echo "Error: Invalid option"
                exit 1
                ;;
        esac
    done
}

prepare_python_environment() {
    log_verbose "-- Preparing python environment for test..."
    "${PYTHON_BIN}" -m venv --clear "${TEST_VENV}"

    # shellcheck disable=SC1091
    source "${TEST_VENV}/bin/activate"

    log_verbose "---- Python version = $(python -V)"
    pip install -U "${TEST_REQUIREMENTS[@]}"
}

install_editable_package() {
    local package_name="$1"
    local package_path="$2"

    log_verbose "-- Installing ${package_name}..."
    pip install -e "${package_path}"
}

clean_pykeops_cache() {
    log_verbose "-- Cleaning pykeops..."
    "${PYTHON_BIN}" -c 'import pykeops; pykeops.clean_pykeops()'
}

run_pykeops_health_check() {
    echo "-- Running pykeops.check_health()..."
    "${PYTHON_BIN}" -c 'import pykeops; pykeops.check_health()'
}

run_test_suite() {
    local suite_name="$1"
    local suite_path="$2"

    log_verbose "-- Running ${suite_name} tests..."
    pytest -v "${suite_path}"
}

main() {
    parse_options "$@"
    prepare_python_environment
    install_editable_package "keopscore" "${SCRIPT_DIR}/keopscore"
    install_editable_package "pykeops" "${SCRIPT_DIR}/pykeops[test]"
    run_pykeops_health_check
    clean_pykeops_cache
    run_test_suite "keopscore" "keopscore/keopscore/test/"
    run_test_suite "pykeops" "pykeops/pykeops/test/"
}

main "$@"
