#!/bin/bash
wget http://data.mxnet.io/models/imagenet-11k/resnet-152/resnet-152-0000.params -P resnet152/ -q
wget http://data.mxnet.io/models/imagenet-11k/resnet-152/resnet-152-symbol.json -P resnet152/ -q
wget http://data.mxnet.io/models/imagenet-11k/synset.txt -P resnet152/ -q
wget https://s3.amazonaws.com/model-server/inputs/kitten.jpg
mkdir images
for i in {1..41}; do cp kitten.jpg "images/kitten$i.jpg"; done