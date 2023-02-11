#!/bin/bash

# Before an official release, please consider using
# docker builder prune
# to clear the docker cache and make sure that the config works
# with the current version of e.g. the Ubuntu repositories.

KEOPS_VERSION=2.1
GEOMLOSS_VERSION=0.2.5
CUDA_VERSION=11.3
CUDA_CHANNEL=nvidia/label/cuda-11.3.1
PYTORCH_VERSION=1.11.0
TORCHVISION_VERSION=0.12.0
TORCHAUDIO_VERSION=0.11.0
PYTORCH_SCATTER_VERSION=2.0.9

VERSION_TAG=${KEOPS_VERSION}-geomloss${GEOMLOSS_VERSION}-cuda${CUDA_VERSION}-pytorch${PYTORCH_VERSION}

for TARGET in keops keops-doc keops-full
do
    docker build \
    --target ${TARGET} \
    --build-arg KEOPS_VERSION=${KEOPS_VERSION} \
    --build-arg GEOMLOSS_VERSION=${GEOMLOSS_VERSION} \
    --build-arg CUDA_VERSION=${CUDA_VERSION} \
    --build-arg CUDA_CHANNEL=${CUDA_CHANNEL} \
    --build-arg PYTORCH_VERSION=${PYTORCH_VERSION} \
    --build-arg TORCHVISION_VERSION=${TORCHVISION_VERSION} \
    --build-arg TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION} \
    --build-arg PYTORCH_SCATTER_VERSION=${PYTORCH_SCATTER_VERSION} \
    --tag getkeops/${TARGET}:${VERSION_TAG} .

    docker tag getkeops/${TARGET}:${VERSION_TAG} getkeops/${TARGET}:latest
done

# Test your images with e.g.
# docker run getkeops/keops:latest
# And push to Docker Hub:
# docker push getkeops/keops:latest