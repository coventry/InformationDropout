#!/bin/bash

# Prepares ubuntu 16.04 host for training/development

set -ex

if [ -z "$SKIP_NVIDIA_INSTALL" ] ; then
    # Prepare repo for NVIDIA drivers
    NVIDIA_KEY_URL=https://nvidia.github.io/nvidia-docker/gpgkey
    curl -s -L $NVIDIA_KEY_URL | sudo apt-key add -
    NVIDIA_REPO_PATH=nvidia-docker.list
    NVIDIA_REPO_URL=https://nvidia.github.io/nvidia-docker/ubuntu16.04/amd64/
    NVIDIA_REPO_DEST="/etc/apt/sources.list.d/$NVIDIA_REPO_PATH"
    wget -O $NVIDIA_REPO_DEST $NVIDIA_REPO_URL/$NVIDIA_REPO_PATH
    NVIDIA_PACKAGE=nvidia-375
else
    NVIDIA_PACKAGE=''
fi

sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx make p7zip $NVIDIA_PACKAGE

miniconda_path=~/miniconda
mkdir -p $miniconda_path
miniconda_url="https://repo.continuum.io/miniconda"
mc_installer="Miniconda3-latest-Linux-x86_64.sh"
installer_path=$miniconda_path/$mc_installer
wget -q -O $installer_path $miniconda_url/$mc_installer
bash $installer_path -b -f -p $miniconda_path
conda_activate="source $miniconda_path/bin/activate"
for p in bashrc bash_profile ; do  # Ensure conda active in bash shells
    echo "$conda_activate" >> ~/.$p
done
$conda_activate
conda install -y matplotlib scikit-image pytorch attrs cuda80 -c pytorch

if [ -z "$SKIP_NVIDIA_INSTALL" ] ; then
    sudo reboot  # Needed for drivers to take effect.
fi
