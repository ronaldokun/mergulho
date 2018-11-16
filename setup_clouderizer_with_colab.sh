#!/bin/bash
if [ ! -d course-v3 ]; then
        pip3 install pillow==4.1.1 --upgrade
        pip3 install torch_nightly -f https://download.pytorch.org/whl/nightly/cu92/torch_nightly.html --upgrade

        sed -n -e '/^tmpfs \/dev\/shm tmpfs defaults,size=/!p' -e '$atmpfs \/dev\/shm tmpfs defaults,size=1g 0 0' -i /etc/fstab
        mount -o remount /dev/shm

        mkdir -p /content/.torch/
        mkdir -p /content/.fastai/
        ln -s /content/clouderizer/fastai-v3/data/ /content/.fastai/data/ 
        rm -rf /content/sample_data/
        git clone https://github.com/fastai/course-v3
fi

pip3 install fastai --upgrade
cd course-v3
git pull
rm -rf /content/course-v3
