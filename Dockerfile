# syntax=docker/dockerfile:1

# Build this file with:
# ./docker-images.sh

# Base OS:
ARG BASE_IMAGE
# Useful to test support across Python versions:
ARG PYTHON_VERSION

# Numpy:
ARG NUMPY_VERSION

# KeOps version - the most important parameter:
ARG KEOPS_VERSION
# We also include all the libraries hosted on www.kernel-operations.io,
# such as GeomLoss. This is convenient, and has negligible impact
# on the size of the final image. Cuda and PyTorch weigh ~5Gb anyway,
# so there is little point trying to maintain separate images that
# differ by a handful of Python files.
ARG GEOMLOSS_VERSION


# Cuda version for nvcc, etc.:
ARG CUDA_VERSION

# Check https://pytorch.org/ and https://pytorch.org/get-started/previous-versions/
# for compatible version numbers:
ARG PYTORCH_VERSION

# PyTorch scatter (used by the "survival" environment)
# is a dependency that may lag behind PyTorch releases by a few days.
# Please check https://github.com/rusty1s/pytorch_scatter for compatibility info.
#ARG PYTORCH_SCATTER_VERSION=2.1.1

# KeOps relies on PyTest, Hypothesis, Beartype and Jaxtyping for unit tests...
ARG PYTEST_VERSION
ARG HYPOTHESIS_VERSION
ARG JAXTYPING_VERSION
ARG BEARTYPE_VERSION
# and Black for code formatting:
ARG BLACK_VERSION


# First step: 
FROM ${BASE_IMAGE} AS dev-base
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        tmux \
        zip \
        wget \
        libjpeg-dev \
        libpng-dev \
        libxml2-dev \
        openssh-client \
        libssl-dev \
        libcurl4-openssl-dev && \
    rm -rf /var/lib/apt/lists/*
ENV PATH=/home/.local/bin:/opt/conda/bin:$PATH


# Install R and a collection of useful packages.
# This section is very stable, so we include it in the first
# layers of our docker image.
FROM dev-base AS r-env
# N.B.: The explicit non-interactive tag is needed to skip
#       the time zone prompt from the tzdata package.
# N.B.: We install as many packages as possible from the Ubuntu repository
#       to save on compilation times.
# N.B.: We install the latest version of roxygen2 from CRAN, to avoid
#       conflicts with collaborators who may not be working with the
#       exact same version of Ubuntu.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    r-base \
    r-base-dev \
    libtirpc-dev \
    r-cran-survival \
    r-cran-reticulate \
    r-cran-formatr \
    r-cran-tidyverse \
    r-cran-plyr \
    r-cran-matrix \
    r-cran-testthat \
    r-cran-devtools && \
    Rscript -e 'install.packages(c("WCE", "languageserver", "profvis", "tictoc", "roxygen2", "qpdf", "pkgdown", "rmarkdown"))'
# Encoding for R:
ENV LC_ALL=C.UTF-8


FROM r-env AS conda
ARG PYTHON_VERSION
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} \
        conda-build \
        pyyaml \
        numpy=${NUMPY_VERSION} \
        ipython \
        matplotlib \
        ipykernel && \
    /opt/conda/bin/conda clean -ya
# rpy2 on conda is not supported anymore. We install it with pip:
RUN /opt/conda/bin/pip install rpy2
# Switch default matplotlib backend to avoid issues with Qt:
ENV MPLBACKEND=tkagg


# Full CUDA installation, with the headers, from cuda-forge:
FROM conda AS cuda 
ARG CUDA_VERSION 
RUN /opt/conda/bin/conda install cuda=${CUDA_VERSION} -c conda-forge && \
    /opt/conda/bin/conda clean -ya


# Full PyTorch installation:
FROM cuda AS pytorch 
ARG PYTORCH_VERSION
# Now that torch does not support conda anymore, we need to use pip
# that overflows on modest machines...
#RUN mkdir -p ~/pip_cache && \
#    export TMPDIR=~/pip_cache
RUN /opt/conda/bin/pip install \
    torch==${PYTORCH_VERSION} \
    torchvision \
    torchaudio
#    --cache-dir=~/pip_cache


# torch.compile(...) introduced by PyTorch 2.0 links to libcuda.so instead 
# of the usual runtime library libcudart.so. We must therefore export the
# LIBRARY_PATH environment variable to make sure that the linker can find it:
ENV LIBRARY_PATH=/opt/conda/lib/stubs

# KeOps, GeomLoss, black and pytest:
FROM pytorch AS keops
ARG KEOPS_VERSION
ARG GEOMLOSS_VERSION
ARG PYTEST_VERSION
ARG BLACK_VERSION
ARG HYPOTHESIS_VERSION
ARG JAXTYPING_VERSION
ARG BEARTYPE_VERSION
RUN /opt/conda/bin/pip install \
    pykeops==${KEOPS_VERSION} \
    geomloss==${GEOMLOSS_VERSION} \
    pytest==${PYTEST_VERSION} \
    black==${BLACK_VERSION} \
    hypothesis==${HYPOTHESIS_VERSION} \
    jaxtyping==${JAXTYPING_VERSION} \
    beartype==${BEARTYPE_VERSION}

# Work around a compatibility bug for KeOps, caused by the fact that conda 
# currently ships a version of libstdc++ that is slightly older than
# that of Ubuntu 22.04:
#RUN rm /opt/conda/lib/libstdc++.so.6 && \
#    ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /opt/conda/lib/libstdc++.so.6

# Tell KeOps that the CUDA headers can be found in /opt/conda/include/...
ENV CUDA_PATH=/opt/conda/targets/x86_64-linux/
# If survivalGPU, geomloss or keops are mounted in the opt folder, they will override the pip version:
ENV PYTHONPATH=/opt/survivalGPU/:/opt/geomloss/:/opt/keops/pykeops/:/opt/keops/keopscore/:$PYTHONPATH


# Dependencies for the KeOps and GeomLoss documentations:
# N.B.: for interactive matplotlib plots, you may need to set
# export MPLBACKEND=tkagg
# before running your script (there is a bug with the default qtagg backend)
FROM keops AS keops-doc
COPY doc-requirements.txt doc-requirements.txt
RUN /opt/conda/bin/pip install -r doc-requirements.txt

# Super-full environment with optional dependencies:
FROM keops-doc AS keops-full
ARG JAXTYPING_VERSION
# N.B.: GPytorch may mess up with the jaxtyping version, 
# so we install it again:
RUN /opt/conda/bin/pip install \
    jaxtyping==${JAXTYPING_VERSION}


# PyTorch-scatter is a complex dependency:
# it relies on binaries that often lag behind new PyTorch releases
# by a few days/weeks.
#ARG PYTORCH_SCATTER_VERSION
#RUN /opt/conda/bin/pip install torch-scatter -f ${PYTORCH_SCATTER_VERSION}

#RUN /opt/conda/bin/conda install -y -c pyg \
#    pytorch-scatter==${PYTORCH_SCATTER_VERSION} && \
#    /opt/conda/bin/conda clean -ya
