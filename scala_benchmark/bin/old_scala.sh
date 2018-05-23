#!/bin/bash
CURR_DIR=$(cd $(dirname $0); pwd)
PROJ_DIR=$(cd $(dirname $0)/../; pwd)
DATA_DIR=$PROJ_DIR/data

# model dir
MODEL_PATH_PREFIX=$1
# input image
INPUT_IMG=$2
# which input image dir
INPUT_DIR=$3
#synset path
SYNSET=$4
# number of runs
NUM_RUN=$5

CLASSPATH=$CLASSPATH:/home/ubuntu/incubator-mxnet/scala-package/assembly/linux-x86_64-cpu/target/*:$CLASSPATH:/home/ubuntu/BenchmarkAI_Scala/target/*:$CLASSPATH:/home/ubuntu/BenchmarkAI_Scala/target/classes/lib/*
java -Xmx16G -cp $CLASSPATH \
  mxnet.example.dataiter.BenchmarkResnet \
    	--model-path-prefix $MODEL_PATH_PREFIX \
	--input-image $INPUT_IMG \
	--input-dir $INPUT_DIR \
	--synset $SYNSET \
	--num-run $NUM_RUN

