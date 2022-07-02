# syntax=docker/dockerfile:1

# Build this file with e.g.
#
# docker build \
# --target pykeops \
# --tag getkeops/keops:2.1-cuda11.3-pytorch1.11-pykeops \
# --no-cache .
#
# or
#
# docker build \
# --target doc \
# --build-arg KEOPS_VERSION=2.1 \
# --build-arg PYTORCH_VERSION=1.12.0 \
# --build-arg TORCHVISION_VERSION=0.13.0 \
# --build-arg TORCHAUDIO_VERSION=0.12.0 \
# --tag getkeops/keops:2.1-cuda11.3-pytorch1.12-doc \
# --no-cache .

ARG BASE_IMAGE=ubuntu:22.04
ARG PYTHON_VERSION=3.8

# Cuda version for the Pytorch install:
ARG CUDA_VERSION=11.3
# Cuda version for the "full" install with development
# headers, nvcc, etc.:
ARG CUDA_CHANNEL=nvidia/label/cuda-11.3.1

# Check https://pytorch.org/ and https://pytorch.org/get-started/previous-versions/
# for compatible version numbers:
ARG PYTORCH_VERSION=1.11.0
ARG TORCHVISION_VERSION=0.12.0
ARG TORCHAUDIO_VERSION=0.11.0

ARG KEOPS_VERSION=2.1
ARG PYTEST_VERSION=7.1.2
ARG BLACK_VERSION=22.6.0


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


FROM dev-base AS conda
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


# KeOps, black and pytest:
FROM pytorch AS pykeops
ARG KEOPS_VERSION
ARG PYTEST_VERSION
ARG BLACK_VERSION
RUN /opt/conda/bin/pip install pykeops==${KEOPS_VERSION} pytest==${PYTEST_VERSION} black==${BLACK_VERSION}

# Work around a compatibility bug for KeOps, caused by the fact that conda 
# currently ships a version of libstdc++ that is slightly older than
# that of Ubuntu 22.04:
RUN rm /opt/conda/lib/libstdc++.so.6 && \
    ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /opt/conda/lib/libstdc++.so.6

# Tell KeOps that the CUDA headers can be found in /opt/conda/include/...
ENV CUDA_PATH=/opt/conda/
# If geomloss or keops are mounted in the opt folder, they will override the pip version:
ENV PYTHONPATH=/opt/geomloss/:/opt/keops/pykeops/:/opt/keops/keopscore/:$PYTHONPATH


# Dependencies for the KeOps and GeomLoss documentations:
FROM pykeops AS doc
COPY doc-requirements.txt doc-requirements.txt
RUN /opt/conda/bin/pip install -r doc-requirements.txt