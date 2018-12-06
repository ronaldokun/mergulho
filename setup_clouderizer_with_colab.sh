#!/bin/bash
pip3 install --upgrade pip
pip3 install dataclasses
pip3 uninstall numpy torchvision_nightly torch torch_nightly -y
pip3 install torch_nightly -f https://download.pytorch.org/whl/nightly/cu92/torch_nightly.html --upgrade
mkdir -p /content/.torch/
mkdir -p /content/.fastai/
ln -s /content/clouderizer/fastai-v3/data/ /content/.fastai/data 
pip3 install fastai --upgrade
git clone https://github.com/fastai/course-v3.git
mv course-v3 /content/fastai-v3/code/

