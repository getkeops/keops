#!/usr/bin/env bash

set -euo pipefail

readonly PYTHON_BIN="python3"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
readonly TEST_VENV="${SCRIPT_DIR}/.test_venv_pytest"
readonly TEST_REQUIREMENTS=(pip)

KEOPS_VERBOSE_LEVEL=-1
PIP_CONSTRAINT_FILE=""

print_help() {
    cat <<EOF
Test script for keopscore/pykeops packages.

Usage: $0 [option...]

    -h      Print the help
    -v <0|1|2>
            Verbosity level forwarded to KEOPS_VERBOSE and PYKEOPS_VERBOSE
    --pip-constraint <file> 
            Constrain pip installs using the given constraints file.
EOF
}

log_verbose() {
    if [[ "${KEOPS_VERBOSE_LEVEL}" -ge 1 ]]; then
        printf '%b\n' "$1"
    fi
}

parse_options() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h)
                print_help
                exit 0
                ;;
            -v)
                shift
                if [[ $# -eq 0 ]]; then
                    echo "Error: -v requires a level (0, 1 or 2)"
                    exit 1
                fi
                if ! [[ "$1" =~ ^[0-2]$ ]]; then
                    echo "Error: Invalid -v value: $1 (expected 0, 1 or 2)"
                    exit 1
                fi
                KEOPS_VERBOSE_LEVEL="$1"
                ;;
            --pip-constraint)
                shift
                if [[ $# -eq 0 ]]; then
                    echo "Error: --pip-constraint requires a file path"
                    exit 1
                fi
                PIP_CONSTRAINT_FILE="$1"
                ;;
            --pip-constraint=*)
                PIP_CONSTRAINT_FILE="${1#*=}"
                ;;
            *)
                echo "Error: Invalid option: $1"
                exit 1
                ;;
        esac
        shift
    done

    if [[ -n "${PIP_CONSTRAINT_FILE}" && ! -f "${PIP_CONSTRAINT_FILE}" ]]; then
        echo "Error: Constraint file not found: ${PIP_CONSTRAINT_FILE}"
        exit 1
    fi

    log_verbose "## verbose mode (level=${KEOPS_VERBOSE_LEVEL})"
}

run_with_keops_verbose() {
    if [[ "${KEOPS_VERBOSE_LEVEL}" -ne -1 ]]; then
        KEOPS_VERBOSE="${KEOPS_VERBOSE_LEVEL}" PYKEOPS_VERBOSE="${KEOPS_VERBOSE_LEVEL}" "$@"
    else
        "$@"
    fi
}

pip_install() {
    local args=()

    if [[ -n "${PIP_CONSTRAINT_FILE}" ]]; then
        args+=(--constraint "${PIP_CONSTRAINT_FILE}")
    fi

    run_with_keops_verbose "${PYTHON_BIN}" -m pip install "${args[@]}" "$@"
}

prepare_python_environment() {
    log_verbose "-- Preparing python environment for test..."
    "${PYTHON_BIN}" -m venv --clear "${TEST_VENV}"

    # shellcheck disable=SC1091
    source "${TEST_VENV}/bin/activate"

    log_verbose "---- Python version = $(${PYTHON_BIN} -V)"
    pip_install -U "${TEST_REQUIREMENTS[@]}"
}

install_editable_package() {
    local package_name="$1"
    local package_path="$2"

    log_verbose "-- Installing ${package_name}..."
    pip_install -e "${package_path}"
}

run_python_outside_repo() {
    local python_code="$1"
    (
        cd /tmp
        run_with_keops_verbose "${PYTHON_BIN}" -c "${python_code}"
    )
}

clean_pykeops_cache() {
    log_verbose "-- Cleaning pykeops..."
    run_python_outside_repo 'import pykeops; pykeops.clean_pykeops()'
}

run_pykeops_health_check() {
    echo "-- Running pykeops.check_health()..."
    run_python_outside_repo 'import pykeops; pykeops.check_health()'
}

run_test_suite() {
    local suite_name="$1"
    local suite_path="$2"

    log_verbose "-- Running ${suite_name} tests..."
    run_with_keops_verbose pytest -v "${suite_path}"
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
