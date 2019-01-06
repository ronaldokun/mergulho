#!/bin/bash
pip3 install --upgrade pip
pip3 install dataclasses
pip3 install fastai --upgrade
mkdir -p /content/.torch/
mkdir -p /content/.fastai/
mkdir -p /content/clouderizer/fastai-v3/out/models
echo data_path: /content/clouderizer/fastai-v3/data/ > /content/.fastai/config.yml
echo model_path: /content/clouderizer/fastai-v3/out/ >> /content/.fastai/config.yml



