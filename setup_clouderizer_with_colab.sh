#!/bin/bash
pip3 install pillow==4.1.1 --upgrade
pip3 install numpy torchvision_nightly --upgrade
pip3 install torch_nightly -f https://download.pytorch.org/whl/nightly/cu92/torch_nightly.html --upgrade
mkdir -p /content/.torch/
mkdir -p /content/.fastai/
ln -s /content/clouderizer/fastai-v3/data/ /content/.fastai/data 
pip3 install fastai --upgrade

