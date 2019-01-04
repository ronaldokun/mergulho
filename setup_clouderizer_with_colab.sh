#!/bin/bash
pip3 install --upgrade pip
pip3 install dataclasses
pip3 install fastai --upgrade
pip3 install jupyterlab-discovery
mkdir -p /content/.torch/
mkdir -p /content/.fastai/
mkdir -p /content/clouderizer/fastai-v3/out/models
cat data_path: /content/clouderizer/fastai-v3/data/ > /content/.fastai/config.yml
cat model_path: /content/clouderizer/fastai-v3/out/ >> /content/.fastai/config.yml



