#!/bin/bash
pip3 install --upgrade pip
pip3 install dataclasses
pip3 install fastai --upgrade
mkdir -p /content/.torch/
mkdir -p /content/.fastai/
echo data_path: /content/clouderizer/fastai-v1/data > ~/.fastai/config.yml
echo model_path: /content/clouderizer/fastai-v1/out >> ~/.fastai/config.yml

