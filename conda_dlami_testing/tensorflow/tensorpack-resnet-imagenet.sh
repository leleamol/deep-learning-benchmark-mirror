git clone https://github.com/jmazanec15/tensorpack.git
cp -r tensorpack/tensorpack tensorpack/examples/ResNet
pip install tqdm
pip install subprocess32
pip install pyarrow
pip install functools32
pip install zmq
pip install tabulate
pip install opencv-python
pip install termcolor
apt-get update -y && apt-get install -y libsm6 libxext6
apt-get install -y libfontconfig1 libxrender1
python tensorpack/examples/ResNet/imagenet-resnet.py --data ~/imagenet --gpu 0,1,2,3,4,5,6,7 -d 50 --data_format 'NHWC' --batch 64 --epochs 10
