#!/bin/bash

set -ex

# Docker repo
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
     "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update

# NVIDIA drivers' repo
NVIDIA_KEY_URL=https://nvidia.github.io/nvidia-docker/gpgkey
curl -s -L $NVIDIA_KEY_URL | sudo apt-key add -
NVIDIA_REPO_PATH=nvidia-docker.list
NVIDIA_REPO_URL=https://nvidia.github.io/nvidia-docker/ubuntu16.04/amd64/
NVIDIA_REPO_DEST="/etc/apt/sources.list.d/$NVIDIA_REPO_PATH"
wget -O $NVIDIA_REPO_DEST $NVIDIA_REPO_URL/$NVIDIA_REPO_PATH

# nvidia-docker's repo
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
    sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu16.04/amd64/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y docker-ce nvidia-375 nvidia-docker2

sudo reboot
