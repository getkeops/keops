# syntax=docker/dockerfile:1

# Build this file with e.g.
#
# docker build \
# --target full \
# --build-arg KEOPS_VERSION=2.1 \
# --build-arg GEOMLOSS_VERSION=0.2.5 \
# --build-arg CUDA_VERSION=11.3 \
# --build-arg CUDA_CHANNEL=nvidia/label/cuda-11.3.1
# --build-arg PYTORCH_VERSION=1.11.0 \
# --build-arg TORCHVISION_VERSION=0.13.0 \
# --build-arg TORCHAUDIO_VERSION=0.12.0 \
# --build-arg PYTORCH_SCATTER_VERSION=2.0.9 \
# --tag getkeops/keops:2.1-geomloss0.2.5-cuda11.3-pytorch1.11-full \
# --no-cache .

# KeOps version - the most important parameter:
ARG KEOPS_VERSION=2.1
# We also include all the libraries hosted on www.kernel-operations.io,
# such as GeomLoss. This is convenient, and has negligible impact
# on the size of the final image. Cuda and PyTorch weigh ~5Gb anyway,
# so there is little point trying to maintain separate images that
# differ by a handful of Python files.
ARG GEOMLOSS_VERSION=0.2.5


# Base OS:
ARG BASE_IMAGE=ubuntu:22.04
# Useful to test support across Python versions:
ARG PYTHON_VERSION=3.8

# Cuda version for the Pytorch install:
ARG CUDA_VERSION=11.3
# Cuda version for the "full" install with development headers, nvcc, etc.:
ARG CUDA_CHANNEL=nvidia/label/cuda-11.3.1

# Check https://pytorch.org/ and https://pytorch.org/get-started/previous-versions/
# for compatible version numbers:
ARG PYTORCH_VERSION=1.11.0
ARG TORCHVISION_VERSION=0.12.0
ARG TORCHAUDIO_VERSION=0.11.0

# PyTorch scatter (used by the "survival" environment)
# is a dependency that may lag behind PyTorch releases by a few days.
# Please check https://github.com/rusty1s/pytorch_scatter for compatibility info.
ARG PYTORCH_SCATTER_VERSION=2.0.9

# KeOps relies on PyTest for unit tests, and Black for code formatting:
ARG PYTEST_VERSION=7.1.2
ARG BLACK_VERSION=22.6.0


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
ENV PATH=/opt/conda/bin:$PATH


# Install R and a collection of useful packages.
# This section is very stable, so we include it in the first
# layers of our docker image.
FROM dev-base AS r-env
# N.B.: The explicit non-interactive tag is needed to skip
#       the time zone prompt from the tzdata package.
# N.B.: We install as many packages as possible from the Ubuntu repository
#       to save on compilation times.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    r-base \
    r-base-dev \
    r-cran-survival \
    r-cran-rmarkdown \
    r-cran-reticulate \
    r-cran-formatr \
    r-cran-tidyverse \
    r-cran-plyr \
    r-cran-matrix \
    r-cran-testthat && \
    Rscript -e 'install.packages(c("WCE", "languageserver", "profvis", "tictoc"))'
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
        numpy \
        ipython \
        matplotlib \
        ipykernel && \
    /opt/conda/bin/conda clean -ya


# Full CUDA installation, with the headers, from the official Nvidia repository:
FROM conda AS cuda 
ARG CUDA_CHANNEL 
RUN /opt/conda/bin/conda install -y -c "${CUDA_CHANNEL}" cuda && \
    /opt/conda/bin/conda clean -ya


# Full PyTorch installation:
FROM cuda AS pytorch 
ARG PYTHON_VERSION
ARG CUDA_VERSION
ARG PYTORCH_VERSION 
ARG TORCHVISION_VERSION
ARG TORCHAUDIO_VERSION
ENV CONDA_OVERRIDE_CUDA=${CUDA_VERSION}
RUN /opt/conda/bin/conda install -y -c pytorch \
        pytorch==${PYTORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION} \
        torchaudio==${TORCHAUDIO_VERSION} \
        python=${PYTHON_VERSION} \
        cudatoolkit=${CUDA_VERSION} && \
    /opt/conda/bin/conda clean -ya


# KeOps, GeomLoss, black and pytest:
FROM pytorch AS keops
ARG KEOPS_VERSION
ARG GEOMLOSS_VERSION
ARG PYTEST_VERSION
ARG BLACK_VERSION
RUN /opt/conda/bin/pip install \
    pykeops==${KEOPS_VERSION} \
    geomloss==${GEOMLOSS_VERSION} \
    pytest==${PYTEST_VERSION} \
    black==${BLACK_VERSION}

# Work around a compatibility bug for KeOps, caused by the fact that conda 
# currently ships a version of libstdc++ that is slightly older than
# that of Ubuntu 22.04:
RUN rm /opt/conda/lib/libstdc++.so.6 && \
    ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /opt/conda/lib/libstdc++.so.6

# Tell KeOps that the CUDA headers can be found in /opt/conda/include/...
ENV CUDA_PATH=/opt/conda/
# If survival-GPU, geomloss or keops are mounted in the opt folder, they will override the pip version:
ENV PYTHONPATH=/opt/survival-GPU/:/opt/geomloss/:/opt/keops/pykeops/:/opt/keops/keopscore/:$PYTHONPATH


# Dependencies for the KeOps and GeomLoss documentations:
FROM keops AS keops-doc
COPY doc-requirements.txt doc-requirements.txt
RUN /opt/conda/bin/pip install -r doc-requirements.txt


# Super-full environment with optional dependencies:
FROM keops-doc as keops-full
# PyTorch-scatter is a complex dependency:
# it relies on binaries that often lag behind new PyTorch releases
# by a few days/weeks.
ARG PYTORCH_SCATTER_VERSION
RUN /opt/conda/bin/conda install -y -c pyg \
    pytorch-scatter==${PYTORCH_SCATTER_VERSION} && \
    /opt/conda/bin/conda clean -ya
