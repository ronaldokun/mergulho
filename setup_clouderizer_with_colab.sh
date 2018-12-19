#!/bin/bash
pip3 install --upgrade pip
pip3 install dataclasses
pip3 uninstall numpy torchvision_nightly torch torch_nightly -y
pip3 install torch_nightly -f https://download.pytorch.org/whl/nightly/cu92/torch_nightly.html --upgrade
mkdir -p /content/.torch/
mkdir -p /content/.fastai/
echo data_path: /content/clouderizer/fastai-v1/data > ~/.fastai/config.yml
echo model_path: /content/clouderizer/fastai-v1/out >> ~/.fastai/config.yml
pip3 install fastai --upgrade

