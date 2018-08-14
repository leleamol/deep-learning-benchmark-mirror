# Based on https://github.com/apache/incubator-mxnet/blob/master/example/image-classification/train_imagenet.py

from mxnet.gluon import nn


class BasicBlock(nn.HybridBlock):
    """
    Pre-activation Residual Block
    2 convolution layers
    """
    def __init__(self, channels, stride=1, dim_match=True):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.dim_match = dim_match
        with self.name_scope():
            self.bn1 = nn.BatchNorm(epsilon=2e-5)
            self.conv1 = nn.Conv2D(channels=channels, kernel_size=3, padding=1, strides=stride, use_bias=False)
            self.bn2 = nn.BatchNorm(epsilon=2e-5)
            self.conv2 = nn.Conv2D(channels=channels, kernel_size=3, padding=1, strides=1, use_bias=False)
            if not self.dim_match:
                self.conv3 = nn.Conv2D(channels=channels, kernel_size=1, padding=0, strides=stride, use_bias=False)

    def hybrid_forward(self, F, x):
        act1 = F.relu(self.bn1(x))
        act2 = F.relu(self.bn2(self.conv1(act1)))
        out = self.conv2(act2)
        if self.dim_match:
            shortcut = x
        else:
            shortcut = self.conv3(act1)
        return out + shortcut


class resnet56Basic(nn.HybridBlock):
    def __init__(self, num_classes):
        super(resnet56Basic, self).__init__()
        with self.name_scope():
            net = self.net = nn.HybridSequential()
            # data normalization
            net.add(nn.BatchNorm(epsilon=2e-5, scale=True))
            # pre-stage
            net.add(nn.Conv2D(channels=16, kernel_size=3, strides=1, padding=1, use_bias=False))
            # Stage 1 (9 total)
            net.add(BasicBlock(16, stride=1, dim_match=False))
            for _ in range(8):
                net.add(BasicBlock(16, stride=1, dim_match=True))
            # Stage 2 (9 total)
            net.add(BasicBlock(32, stride=2, dim_match=False))
            for _ in range(8):
                net.add(BasicBlock(32, stride=1, dim_match=True))
            # Stage 3 (9 total)
            net.add(BasicBlock(64, stride=2, dim_match=False))
            for _ in range(8):
                net.add(BasicBlock(64, stride=1, dim_match=True))
            # post-stage (required as using pre-activation blocks)
            net.add(nn.BatchNorm(epsilon=2e-5))
            net.add(nn.Activation('relu'))
            # net.add(nn.AvgPool2D(8)) # should be identical to global avg pool as feature map would be 8x8 at this stage
            net.add(nn.GlobalAvgPool2D())
            net.add(nn.Dense(num_classes))

    def hybrid_forward(self, F, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
        return out