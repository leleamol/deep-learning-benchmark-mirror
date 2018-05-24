#!/bin/bash
# For CPU Test
sudo apt-get update
sudo apt-get install -y maven
git clone --recursive https://github.com/apache/incubator-mxnet $HOME/incubator-mxnet
cp -R scala_benchmark/ $HOME/
cd $HOME/incubator-mxnet
make -j4
make scalapkg
make scalatest
cd ../scala_benchmark
mvn package
bash bin/download.sh
bash bin/new_scala.sh resnet152/resnet-152 kitten.jpg images/ 1