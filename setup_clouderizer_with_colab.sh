#!/bin/bash
if [ ! -d course-v3 ]; then
        pip3 install pillow==4.1.1 --upgrade
        pip3 install torch_nightly -f https://download.pytorch.org/whl/nightly/cu92/torch_nightly.html --upgrade

        sed -n -e '/^tmpfs \/dev\/shm tmpfs defaults,size=/!p' -e '$atmpfs \/dev\/shm tmpfs defaults,size=1g 0 0' -i /etc/fstab
        mount -o remount /dev/shm

        mkdir -p /content/.torch/models
        mkdir -p /content/.fastai/data
        ln -s /content/.torch/models /content/fastai-v3/data
        ln -s /content/.fastai/data /content/fastai-v3/data
        rm -rf /content/sample_data/
        git clone https://github.com/fastai/course-v3
fi

pip3 install fastai --upgrade
cd course-v3
git pull
mv /content/course-v3 /content/clouderizer/fastai-v3/
