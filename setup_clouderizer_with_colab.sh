#!/bin/bash
# Install fastai, jupyter, jupylab and dependencies ( The docker container in clouderizer is inside another Docker Container [ the Colab or Kaggle One ] so no use of conda )

pip3 install --upgrade pip
pip3 install dataclasses
pip3 install fastai --upgrade
pip3 install jupyter --upgrade
pip3 install jupyterlab --upgrade
pip3 install jupyter_nbextensions_configurator
pip3 install jupyter_contrib_nbextensions
pip3 install nbconvert
pip3 install fire

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

# Config the automatic generation of a python script as we run a notebook 
jupyter notebook --generate-config

echo c.NotebookApp.contents_manager_class = "jupytext.TextFileContentsManager" > /content/.jupyter/jupyter_notebook_config.py

echo c.ContentsManager.preferred_jupytext_formats_save = "py:percent" >> /content/.jupyter/jupyter_notebook_config.py
