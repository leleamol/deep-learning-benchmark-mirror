git clone https://github.com/jmazanec15/tensorpack.git
cp -r tensorpack/tensorpack tensorpack/examples/ResNet
pip2 install tqdm
pip2 install subprocess32
pip2 install pyarrow
pip2 install functools32
pip2 install zmq
pip2 install tabulate
pip2 install opencv-python
apt-get update -y && apt-get install -y libsm6 libxext6
apt-get install -y libfontconfig1 libxrender1
python2 tensorpack/examples/ResNet/imagenet-resnet.py --data ~/imagenet --gpu 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 -d 50 --data_format 'NHWC' --batch 64
