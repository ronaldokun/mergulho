#!/bin/bash
pip3 install --upgrade pip
pip3 install dataclasses
pip3 install fastai --upgrade
pip3 install jupyter --upgrade 
pip3 install jupyter_nbextensions_configurator --upgrade 
pip3 install jupyter_contrib_nbextensions --upgrade
jupyter contrib nbextension install
jupyter nbextensions_configurator enable
pip3 install --no-deps pretrainedmodels

mkdir -p /content/.torch/
mkdir -p /content/.fastai/
mkdir -p /content/clouderizer/fastai-1.0/out/models
echo data_path: /content/clouderizer/fastai-1.0/data/ > /content/.fastai/config.yml
echo model_path: /content/clouderizer/fastai-1.0/out/ >> /content/.fastai/config.yml



