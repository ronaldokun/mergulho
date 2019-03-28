#!/bin/bash
pip3 install --upgrade pip
pip3 install dataclasses
pip3 install fastai --upgrade
pip3 install jupyter --upgrade
pip3 install jupyterlab --upgrade
pip3 install jupyter_nbextensions_configurator
pip3 install jupyter_contrib_nbextensions
pip3 install nbconvert
jupyter contrib nbextension install --user
jupyter nbextensions_configurator enable --user

wget https://nodejs.org/dist/v10.15.3/node-v10.15.3-linux-x64.tar.xz --output-document ~/node-v10.15.3-linux-x64.tar.xz
VERSION=v10.15.3
DISTRO=linux-x64
mkdir -p /usr/local/lib/nodejs
tar -xJvf node-$VERSION-$DISTRO.tar.xz -C /usr/local/lib/nodejs
echo PATH=/usr/local/lib/nodejs/node-$VERSION-$DISTRO/bin:$PATH > ~/.profile
. ~/.profile
jupyter labextension install @jupyterlab/toc

pip3 install jupytext --upgrade
pip3 install --no-deps pretrainedmodels

mkdir -p /content/.torch/
mkdir -p /content/.fastai/
mkdir -p /content/clouderizer/fastai-1.0/out/models
echo data_path: /content/clouderizer/fastai-1.0/data/ > /content/.fastai/config.yml
echo model_path: /content/clouderizer/fastai-1.0/out/ >> /content/.fastai/config.yml

jupyter nbextension install --py jupytext --user
jupyter nbextension enable --py jupytext --user

jupyter labextension install jupyterlab-jupytext

jupyter notebook --generate-config

echo c.NotebookApp.contents_manager_class = "jupytext.TextFileContentsManager" > /content/clouderizer/.jupyter/jupyter_notebook_config.py

echo c.ContentsManager.preferred_jupytext_formats_save = "py:percent" >> /content/clouderizer/.jupyter/jupyter_notebook_config.py




