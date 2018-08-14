import cv2
import numpy as np
from matplotlib import pyplot as plt
from multiprocessing import cpu_count
import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms


class ConstantPad():
    def __init__(self, width, value=(0, 0, 0)):
        """
        width: number of padding pixels added to each edge
        value: rgb tuple to use for padding pixels, default is black.
        """
        self.width = width
        self.value = value

    def __call__(self, src):
        """
        data: mx.ndarray in HWC format (uint8)
        return: mx.ndarray in HWC format (uint8)
        """
        src_np = src.asnumpy()
        output = cv2.copyMakeBorder(src_np,
                                    self.width, self.width, self.width, self.width,
                                    cv2.BORDER_CONSTANT, value=self.value)
        return mx.nd.array(output, dtype="uint8")


class RandomCrop():
    def __init__(self, size):
        self.size = size

    def __call__(self, src):
        out, box = mx.image.random_crop(src, size=self.size)
        return out


train_transform = transforms.Compose([
    ConstantPad(width=2),
    RandomCrop(size=(32,32)),
    transforms.RandomFlipLeftRight(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])
])


data = mx.gluon.data.vision.datasets.CIFAR10(train=True).transform_first(train_transform)
batch_size = 128
data_loader = mx.gluon.data.DataLoader(data, batch_size=batch_size, num_workers=cpu_count())


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


class resnet20Basic(nn.HybridBlock):
    def __init__(self, num_classes):
        super(resnet20Basic, self).__init__()
        with self.name_scope():
            net = self.net = nn.HybridSequential()
            # data normalization
            net.add(nn.BatchNorm(epsilon=2e-5, scale=True))
            # pre-stage
            net.add(nn.Conv2D(channels=16, kernel_size=3, strides=1, padding=1, use_bias=False))
            # Stage 1 (3 total)
            net.add(BasicBlock(16, stride=1, dim_match=False))
            for _ in range(2):
                net.add(BasicBlock(16, stride=1, dim_match=True))
            # Stage 2 (3 total)
            net.add(BasicBlock(32, stride=2, dim_match=False))
            for _ in range(2):
                net.add(BasicBlock(32, stride=1, dim_match=True))
            # Stage 3 (3 total)
            net.add(BasicBlock(64, stride=2, dim_match=False))
            for _ in range(2):
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



net = resnet20Basic(num_classes=10)


class ContinuousDataProvider():
    """
    Can provide a continuous stream of data, if continuous argument is set to True when get_batch.
    """
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.reset()

    def reset(self):
        self.data_loader_iter = self.data_loader.__iter__()
        self.data_loader_iter_enum = enumerate(self.data_loader_iter)

    def get_batch(self, continuous=False):
        try:
            return self.data_loader_iter_enum.__next__()
        except StopIteration as e:
            if continuous:
                self.reset()
                return self.data_loader_iter_enum.__next__()
            else:
                raise e

    def close(self):
        self.data_loader_iter.shutdown()


class Learner():
    def __init__(self, net, data_loader, ctx=None):
        self.net = net
        self.data_provider = ContinuousDataProvider(data_loader)
        # Initialize the parameters with Xavier initializer
        param_dict = net.collect_params()
        if ctx is None:
            self.ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
        else:
            self.ctx = ctx
        param_dict.initialize(mx.init.Xavier(), ctx=self.ctx)
        # Use cross entropy loss
        self.loss_fn = mx.gluon.loss.SoftmaxCrossEntropyLoss()
        # Use Adam optimizer
        self.trainer = mx.gluon.Trainer(param_dict, 'adam', {'learning_rate': .001})

    def iteration(self, lr=None, continuous=True):
        if lr and (lr != self.trainer.learning_rate):
            self.trainer.set_learning_rate(lr)
        batch_idx, (data, label) = self.data_provider.get_batch(continuous=continuous)
        # get the images and labels
        data = data.as_in_context(self.ctx)
        label = label.as_in_context(self.ctx)
        # Ask autograd to record the forward pass
        with mx.autograd.record():
            # Run the forward pass
            output = self.net(data)
            # Compute the loss
            loss = self.loss_fn(output, label)
        self.iteration_loss = mx.nd.mean(loss).asscalar()
        # Compute gradients
        loss.backward()
        # Update parameters
        self.trainer.step(data.shape[0])

    def epoch(self, lr=None):
        # restart data loader
        self.data_provider.reset()
        try:
            while True:
                self.iteration(lr=lr, continuous=False)
        except StopIteration:
            pass

    def close(self):
        self.data_provider.close()


learner = Learner(net=net, data_loader=data_loader)

for iteration_idx in range(500):
    learner.iteration(lr=0.1)
    print(iteration_idx)
    print(learner.iteration_loss)

learner.close()








