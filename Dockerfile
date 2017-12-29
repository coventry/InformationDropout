FROM ubuntu:xenial

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y sudo curl wget bzip2

# Drivers need to be on the host
ENV SKIP_NVIDIA_INSTALL=True
COPY ./preprocessing/install.sh /src/preprocessing/install.sh
WORKDIR /src
RUN bash ./preprocessing/install.sh

COPY . /src
RUN . ~/miniconda/bin/activate && make -f ./preprocessing/Makefile data
CMD . ~/miniconda/bin/activate && python bentes.py
