This is trains a convolutional neural net to distinguish ships from icebergs,
using the data from the [Statoil Iceberg Kaggle Challenge](
https://www.kaggle.com/c/statoil-iceberg-classifier-challenge).

Currently, it does a pretty bad job of it. See the [Background section](
#background)

# Extracting and viewing the data

## Requirements

On a contemporary ubuntu machine, `sudo bash preprocessing/install.sh` should
install all requirements. The install script and the following instructions have
been tested on an AWS EC2 g2.2xlarge instance running Ubuntu 16.04, with 20GB of
disk space.

## Caution

__The installation script makes invasive changes to the target system.__ Use at
your own risk! I have not bothered to isolate its effects during development,
because I've been working on throwaway EC2 instances.

The script attempts to
- install NVIDIA drivers, `apt` packages, and Anaconda;
- mess with your dotfiles and install some `conda` packages;
- and finally, trigger a reboot.

The [included Dockerfile]( ./Dockerfile) allows you to run this code under
[nvidia-docker]( https://github.com/NVIDIA/nvidia-docker) after placing a copy
of `train.json.7z` (see [below](#steps)) in this directory and running `docker
build -t statoil .; nvidia-docker run statoil`. You can install `nvidia-docker`
on ubuntu if you wish using [`sudo bash ./preprocessing/docker-install.sh`](
./preprocessing/docker-install.sh), which also installs `docker` and NVIDIA
drivers on the target system if necessary. Again, invasive changes; use at your
own risk.

## Steps

1. Download `train.json.7z` from the [data page](
https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/data) to this
directory.

2. Run the command `make -f ./preprocessing/Makefile data` to extract the data.

This will create directories `./artifacts/train` and `./artifacts/images`. These
contain one file per example. See the `preprocessing/extract.py` docstring for
details on the formats of the files and filenames.

# Running the training procedure

`python ./bentes.py`

# Background

I've been reading [Achille & Soatto's 2016 paper "Information
Dropout."](https://arxiv.org/pdf/1611.01353) and sequels with great interest and
enthusiasm. It's a very appealing theory for why Deep Learning works. I'm having
trouble generalizing their results to other data. I was wondering how other
people have fared with it, and also looking for feedback on my implementation.

I'm trying to apply it to the [Statoil Kaggle challenge](
https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/data) because the
[images have a lot of background noise](
https://storage.googleapis.com/kaggle-media/competitions/statoil/8ZkRcp4.png),
reminiscent of the "cluttered MNIST" problem Achille & Soatto test on in
"Information Dropout."
