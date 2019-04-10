#!/bin/bash
# Install fastai, jupyter, jupylab and dependencies ( The docker container in clouderizer is inside another Docker Container [ the Colab or Kaggle One ] so no use of conda )

pip3 install --upgrade pip dataclasses fastai jupyter ipython jupyterlab jupyter_nbextensions_configurator jupyter_contrib_nbextensions nbconvert fire

# Fix issue described here: https://github.com/jupyter/jupyter/issues/270#issuecomment-322969531
ipython3 kernel install

# Install the jupyter notebook extensions and configurator
jupyter contrib nbextension install --user
jupyter nbextensions_configurator enable --user

# install nodejs dependency for the Jupyterlab Extensions
VERSION=v10.15.3
DISTRO=linux-x64
wget https://nodejs.org/dist/$VERSION/node-$VERSION-$DISTRO.tar.xz --output-document ~/node-$VERSION-$DISTRO.tar.xz
mkdir -p /usr/local/lib/nodejs
tar -xJvf node-$VERSION-$DISTRO.tar.xz -C /usr/local/lib/nodejs
echo PATH=/usr/local/lib/nodejs/node-$VERSION-$DISTRO/bin:$PATH > ~/.profile
. ~/.profile
rm -rf ~/node-$VERSION-$DISTRO.tar.xz

# Install the Jupyterlab extensions
jupyter labextension install @jupyterlab/toc

# Install Jupytext ( Paralel Editing of Jupyter Notebooks as scripts )
pip3 install jupytext --upgrade

# Install the pretrainedmodels
pip3 install --no-deps pretrainedmodels

# Setup the folders
mkdir -p /content/.torch/
mkdir -p /content/.fastai/
mkdir -p /content/clouderizer/fastai-1.0/out/models
echo data_path: /content/clouderizer/fastai-1.0/data/ > /content/.fastai/config.yml
echo model_path: /content/clouderizer/fastai-1.0/out/ >> /content/.fastai/config.yml
