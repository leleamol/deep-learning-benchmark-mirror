# DAWNBench CIFAR-10 MXNet

An implementation of Resnet-164 (without Bottleneck) in MXNet for DAWNBench.

Model is written using Gluon's `HybridBlock`s, and then trained using Gluon and Module APIs.

# Requirements

* MXNet
* MXBoard (optionally)

# Instructions

1) Start AWS EC2 instance using DLAMI (ideally a p3.2xlarge instance)
2) Clone repository, and `cd` into directory.
3) Activate MXNet environment: `source activate mxnet_p36`
4) `python experiments/resnet164_basic_gluon.py --gpu-idxs=0`
5) Open separate terminal/screen for TensorBoard server.
6) `tensorboard --logdir ./logs/tensorboard/`

# BenchmarkAI

```
cd /Users/tholane/Documents/AWSDlBench/src/AWSDlBench/src/aws_dl_bench
python dlbench.py run dawnbench_test_gluon /Users/tholane/mxnet/deep-learning-benchmark-mirror/dawnbench/experiments/resnet164_basic_gluon.json benchmarkai
python dlbench.py run dawnbench_test_gluon_hybrid /Users/tholane/mxnet/deep-learning-benchmark-mirror/dawnbench/experiments/resnet164_basic_gluon_hybrid.json benchmarkai
python dlbench.py run dawnbench_test_module /Users/tholane/mxnet/deep-learning-benchmark-mirror/dawnbench/experiments/resnet164_basic_module.json benchmarkai
python dlbench.py checkstatus dawnbench_test_gluon benchmarkai
python dlbench.py checkstatus dawnbench_test_gluon_hybrid benchmarkai
python dlbench.py checkstatus dawnbench_test_module benchmarkai

python dlbench.py events --create /Users/tholane/mxnet/deep-learning-benchmark-mirror/dawnbench/experiments/resnet164_basic_gluon_nightly.json benchmarkai
python dlbench.py events --create /Users/tholane/mxnet/deep-learning-benchmark-mirror/dawnbench/experiments/resnet164_basic_gluon_hybrid_nightly.json benchmarkai
python dlbench.py events --create /Users/tholane/mxnet/deep-learning-benchmark-mirror/dawnbench/experiments/resnet164_basic_module_nightly.json benchmarkai
```
