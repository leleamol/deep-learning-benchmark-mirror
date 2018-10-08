#!/usr/bin/env bash
set -ex

cd /incubator-mxnet/scala-package/examples/scripts/infer/images/
max=1000
for i in `seq 2 $max`
do
    cp kitten.jpg kitten$i.jpg
done

echo "Done copying"