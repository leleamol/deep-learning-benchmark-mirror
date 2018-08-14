"""
MXNet implementation of Stanford Future Data Systems' Tensorflow model,
that was used for DAWNBench submission.
Using num_residual_units = 27, without bottleneck (i.e. basic block).
https://github.com/stanford-futuredata/dawn-bench-models/blob/master/tensorflow/CIFAR10/resnet/resnet_model.py
"""

# Note this is NOT Resnet-164
# Uses basic block instead of bottleneck (as is used in Resnet-164).
# Uses 27 blocks in each stage instead of 18
# Uses 170 layers as a result.
# Uses AvgPooling in each block.


from mxnet.gluon import nn


class Basic(nn.HybridBlock):
    def __init__(self, in_filter, out_filter, stride, activate_before_residual=False):
        super(Basic, self).__init__()
        self.in_filter = in_filter
        self.out_filter = out_filter
        self.activate_before_residual = activate_before_residual
        with self.name_scope():
            self.bn1 = nn.BatchNorm()
            self.conv1 = nn.Conv2D(channels=out_filter,
                                   kernel_size=3,
                                   padding=1,
                                   strides=stride,
                                   use_bias=False)
            self.bn2 = nn.BatchNorm()
            self.conv2 = nn.Conv2D(channels=out_filter,
                                   kernel_size=3,
                                   padding=1,
                                   strides=1,
                                   use_bias=False)

            self.avgpool = nn.AvgPool2D(pool_size=stride, strides=stride, padding=0)


    def hybrid_forward(self, F, x):
        if self.activate_before_residual:
            x = F.LeakyReLU(self.bn1(x), slope=0.1)
            orig_x = x
        else:
            orig_x = x
            x = F.LeakyReLU(self.bn1(x), slope=0.1)

        x = self.conv1(x)
        x = self.conv2(F.LeakyReLU(self.bn2(x), slope=0.1))

        if self.in_filter != self.out_filter:
            orig_x = self.avgpool(orig_x)
            # Workaround due to "Current implementation expects padding on the first two axes to be zero."
            orig_x = orig_x.transpose()
            orig_x = F.pad(orig_x,
                           mode="constant", constant_value=0,
                           pad_width=(0,0,
                                      0,0,
                                      (self.out_filter - self.in_filter) // 2,
                                      (self.out_filter - self.in_filter) // 2,
                                      0,0))
            orig_x = orig_x.transpose()

        x = x + orig_x
        return x


class PreActResnet164Basic(nn.HybridBlock):
    def __init__(self, num_classes):
        super(PreActResnet164Basic, self).__init__()
        with self.name_scope():
            net = self.net = nn.HybridSequential()
            # block 1
            net.add(nn.Conv2D(16, 3, 1, 1, use_bias=False))
            # block 2
            net.add(Basic(in_filter=16, out_filter=16, stride=1, activate_before_residual=True))
            for _ in range(27):
                net.add(Basic(in_filter=16, out_filter=16, stride=1))
            # block 3
            net.add(Basic(in_filter=16, out_filter=32, stride=2))
            for _ in range(27):
                net.add(Basic(in_filter=32, out_filter=32, stride=1))
            # block 4
            net.add(Basic(in_filter=32, out_filter=64, stride=2))
            for _ in range(27):
                net.add(Basic(in_filter=64, out_filter=64, stride=1))
            # block 5
            net.add(nn.BatchNorm())
            net.add(nn.LeakyReLU(alpha=0.1))
            net.add(nn.GlobalAvgPool2D())
            net.add(nn.Dense(num_classes))

    def hybrid_forward(self, F, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
        return out