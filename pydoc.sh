#!/usr/bin/env bash

set -euo pipefail

readonly PYTHON_BIN="python3"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
readonly DOC_VENV="${SCRIPT_DIR}/.doc_venv"
readonly DOC_REQUIREMENTS=(pip)
readonly DOC_DIR="${SCRIPT_DIR}/doc"

FIX_LINKS=0
NO_PLOT=0
DOC_JOBS=""

print_help() {
    cat <<EOF
Build the documentation and optionally fix generated links.

Usage: $0 [option...]

   -h     Print the help
   -j N   Use N parallel workers for Sphinx and Sphinx-Gallery
   -l     Fix generated documentation links
   -n     Skip plot generation (make html-noplot)
EOF
}

log_step() {
    printf '%s\n' "$1"
}

parse_options() {
    while getopts ":hj:ln" option; do
        case "${option}" in
            h)
                print_help
                exit 0
                ;;
            j)
                if ! [[ "${OPTARG}" =~ ^[1-9][0-9]*$ ]]; then
                    echo "Error: -j expects a positive integer"
                    exit 1
                fi
                DOC_JOBS="${OPTARG}"
                ;;
            l)
                FIX_LINKS=1
                ;;
            n)
                NO_PLOT=1
                ;;
            :)
                echo "Error: Option -${OPTARG} requires an argument"
                exit 1
                ;;
            \?)
                echo "Error: Invalid option"
                exit 1
                ;;
        esac
    done
}

prepare_python_environment() {
    log_step "-- Preparing python environment for doc build..."
    "${PYTHON_BIN}" -m venv --clear "${DOC_VENV}"

    # shellcheck disable=SC1091
    source "${DOC_VENV}/bin/activate"

    log_step "---- Python version = $(${PYTHON_BIN} -V)"
    "${PYTHON_BIN}" -m pip install -U "${DOC_REQUIREMENTS[@]}"
}

install_editable_package() {
    local package_name="$1"
    local package_path="$2"

    log_step "-- Installing ${package_name}..."
    "${PYTHON_BIN}" -m pip install -e "${package_path}"
}

build_doc() {
    local make_target
    local make_args=()

    log_step ""
    log_step "----------------------"
    log_step "   Building the doc"
    log_step "----------------------"
    log_step ""

    if [[ -n "${DOC_JOBS}" ]]; then
        log_step "-- Using ${DOC_JOBS} parallel workers"
        # The SPHINXOPTS -j currently only affects sphinx-build -j, not gallery workers
        # Add an explicit env to be read in the conf.py
        make_args+=("SPHINX_GALLERY_JOBS=${DOC_JOBS}" "SPHINXOPTS=-j ${DOC_JOBS}")
    fi

    pushd "${DOC_DIR}" >/dev/null
    make clean
    if [[ "${NO_PLOT}" -eq 1 ]]; then
        make_target="html-noplot"
    else
        make_target="html"
    fi
    make "${make_args[@]}" "${make_target}"
    popd >/dev/null
}

fix_doc_links() {
    if [[ "${FIX_LINKS}" -ne 1 ]]; then
        return
    fi

    log_step ""
    log_step "----------------------"
    log_step "   Fixing doc links"
    log_step "----------------------"
    log_step ""

    pushd "${DOC_DIR}" >/dev/null
    find . -path "*_auto_*" -name "plot_*.html" -exec \
        sed -i "s/doc\/_auto_\(.*\)rst/pykeops\/pykeops\/\1py/" {} \;
    find . -path "*_auto_*" -name "index.html" -exec \
        sed -i "s/doc\/_auto_\(.*\)\/index\.rst/pykeops\/pykeops\/\1\//" {} \;
    popd >/dev/null
}

main() {
    parse_options "$@"
    prepare_python_environment
    install_editable_package "keopscore" "${SCRIPT_DIR}/keopscore"
    install_editable_package "pykeops" "${SCRIPT_DIR}/pykeops[full]"
    build_doc
    fix_doc_links
}

main "$@"
