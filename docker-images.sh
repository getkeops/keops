#!/bin/bash

# Before an official release, please consider using
# docker builder prune
# to clear the docker cache and make sure that the config works
# with the current version of e.g. the Ubuntu repositories.

# Up to date as of Monday, Jan. 7th, 2025:
PYTHON_VERSION=3.12
NUMPY_VERSION=2.2.1
KEOPS_VERSION=2.2.3
GEOMLOSS_VERSION=0.2.6
CUDA_VERSION=12.4
CUDA_CHANNEL=nvidia/label/cuda-12.4.1
PYTORCH_VERSION=2.5.1
TORCHVISION_VERSION=0.20.1
TORCHAUDIO_VERSION=2.5.1
PYTORCH_SCATTER_VERSION=https://data.pyg.org/whl/torch-2.5.1+cu124.html
PYTEST_VERSION=8.3.4
HYPOTHESIS_VERSION=6.123.7
JAXTYPING_VERSION=0.2.36
BEARTYPE_VERSION=0.19.0
BLACK_VERSION=24.10.0

VERSION_TAG=${KEOPS_VERSION}-geomloss${GEOMLOSS_VERSION}-cuda${CUDA_VERSION}-pytorch${PYTORCH_VERSION}-python${PYTHON_VERSION}

for TARGET in keops keops-doc keops-full
do
    docker build \
    --target ${TARGET} \
    --build-arg PYTHON_VERSION=${PYTHON_VERSION} \
    --build-arg NUMPY_VERSION=${NUMPY_VERSION} \
    --build-arg KEOPS_VERSION=${KEOPS_VERSION} \
    --build-arg GEOMLOSS_VERSION=${GEOMLOSS_VERSION} \
    --build-arg CUDA_VERSION=${CUDA_VERSION} \
    --build-arg CUDA_CHANNEL=${CUDA_CHANNEL} \
    --build-arg PYTORCH_VERSION=${PYTORCH_VERSION} \
    --build-arg TORCHVISION_VERSION=${TORCHVISION_VERSION} \
    --build-arg TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION} \
    --build-arg PYTORCH_SCATTER_VERSION=${PYTORCH_SCATTER_VERSION} \
    --build-arg PYTEST_VERSION=${PYTEST_VERSION} \
    --build-arg HYPOTHESIS_VERSION=${HYPOTHESIS_VERSION} \
    --build-arg JAXTYPING_VERSION=${JAXTYPING_VERSION} \
    --build-arg BEARTYPE_VERSION=${BEARTYPE_VERSION} \
    --build-arg BLACK_VERSION=${BLACK_VERSION} \
    --tag getkeops/${TARGET}:${VERSION_TAG} .

    docker tag getkeops/${TARGET}:${VERSION_TAG} getkeops/${TARGET}:latest
done

# Test your images with e.g.
# docker run -dit getkeops/keops:latest
# or
# docker run --gpus all -dit getkeops/keops-full:latest
#
# Use
# docker ps
# to get the container id and then
# docker exec -it <container_id> /bin/bash
#
# And push to Docker Hub:
# docker login -u getkeops
# docker push getkeops/keops:latest
# docker push getkeops/keops-doc:latest
# docker push getkeops/keops-full:latest
#
# and do the same thing with the version tag, e.g.
# docker push getkeops/keops-full:2.2-geomloss0.2.6-cuda12.1-pytorch2.1.2-python3.11
#
# To test things with Singularity, you can now follow the instructions at
# http://kernel-operations.io/keops/python/installation.html#using-docker-or-singularity