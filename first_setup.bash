#!/bin/bash -exv
sudo apt update -y && sudo apt upgrade -y
sudo apt install -y python3-pip

sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update -y
#sudo apt install -y python3.7 python3.7-venv
sudo apt install -y python3.6 python3.6-venv

#sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 37
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 36
#sudo update-alternatives --config python3
sudo update-alternatives --set python3 /usr/bin/python3.6


git clone https://github.com/Shogo4402/Mask_RCNN.git
cd Mask_RCNN
pip3 install -r requirements.txt
sudo python3 setup.py install


sudo pip3 install -U numpy
sudo pip3 install -U Cython
sudo apt-get install -y build-essential python3.6-dev
#sudo apt-get install -y build-essential libssl-dev libffi-dev python3.6-dev
#git clone https://github.com/Shogo4402/cocoapi.git
git clone https://github.com/Shogo4402/coco.git
cd coco/PythonAPI
sudo python3 setup.py build_ext install

#ln -s /home/ubuntu/Mask_RCNN/mrcnn/ /home/ubuntu/.local/lib/python3.7/site-packages/
#ln -s /home/ubuntu/Mask_RCNN/cocoapi/PythonAPI/pycocotools/ /home/ubuntu/.local/lib/python3.7/site-packages/
#ln -s /home/ubuntu/Mask_RCNN/samples/coco/ /home/ubuntu/.local/lib/python3.7/site-packages/

 
