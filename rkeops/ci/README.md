# RKeOps Continuous Integration (CI)

## Automatic testing

To run the CI of RKeOps 
```bash
bash run_ci.sh
```

To enable automatic tests to be run on GPU, the environment variable `TEST_GPU` should be defined with the value `1` before running the CI scripts, i.e.
```bash
export TEST_GPU=1
bash run_ci.sh
```

## Details

### Troubleshooting

`devtools` package dependencies
  - `apt-get install build-essential libcurl4-gnutls-dev libxml2-dev libssl-dev` (ubuntu)

Configuration failed to find libgit2 library. Try installing:
  - brew: libgit2 (MacOS)
  - deb: libgit2-dev (Debian, Ubuntu, etc)
  - rpm: libgit2-devel (Fedora, CentOS, RHEL)
  - pacman: libgit2 (arch)

**MacOS**: see [`./prepare_macos_ci.sh`](./prepare_macos_ci.sh)

**Pandoc**: version >= 1.12.3 necessary for Rmarkdown

### Triggering tests and checks

To run the package automatic tests, you must run:
```bash
Rscript run_tests.R
```

To run the R package check (equivalent to `R CMD build` and `R CMD check`), you must run:
```bash
Rscript run_check.R
```

> **Note:** you should run `Rscript prepare_ci.R` before calling `run_tests.R` or `run_check.R` to install dependencies.

## Webpage generation

The RKeOps package webpage is generated with [`pkgdown`](https://pkgdown.r-lib.org/).

You should be able to generate it with:

```bash
cd rkeops/ci
bash build_website.sh
cd ../..
cp -r rkeops/docs/ WEBSITE_ROOT_DIR/rkeops/
```
