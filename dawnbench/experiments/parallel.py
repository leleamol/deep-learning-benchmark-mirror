import mxnet as mx
from mxnet.gluon import nn
from time import sleep


ctx1 = mx.gpu(0)
ctx2 = mx.gpu(1)


batch_size = 100
input_channels = 51
input_dims = 250
filters = 101
kernel_size = 12

data_elems = batch_size * input_channels * input_dims * input_dims
conv_elems = ((input_channels * kernel_size * kernel_size) + 1) * filters

def load_data():
    data = mx.random.uniform(shape=(batch_size, input_channels, input_dims, input_dims))
    # data = data_input.copy()
    data1 = data.as_in_context(ctx1)
    data2 = data.as_in_context(ctx2)
    mx.nd.waitall()
    return data1, data2


def init_layers():
    conv1 = nn.Conv2D(channels=filters, kernel_size=(kernel_size, kernel_size))
    conv1.initialize(ctx=ctx1)
    conv2 = nn.Conv2D(channels=filters, kernel_size=(kernel_size, kernel_size))
    conv2.initialize(ctx=ctx2)
    mx.nd.waitall()
    return conv1, conv2


def forward(data1, data2, conv1, conv2):
    out1 = conv1(data1)
    out2 = conv2(data2)
    mx.nd.waitall()
    return out1, out2


def forward2(data2, conv2):
    out2 = conv2(data2)
    mx.nd.waitall()
    return out2

def forward1(data1, conv1):
    out1 = conv1(data1)
    mx.nd.waitall()
    return out1


if __name__ == "__main__":
    print("Data: {} GB".format(4 * data_elems / (1024 * 1024 * 1024)))
    a, b = load_data()
    print("Weights (& Bias): {} GB".format(4 * conv_elems / (1024 * 1024 * 1024)))
    c1, c2 = init_layers()
    o2 = forward2(b, c2)
    o1 = forward1(a, c1)
    print("Passed once.")
    o2 = forward2(b, c2)
    o1 = forward1(a, c1)
    o1, o2 = forward(a, b, c1, c2)
    print("Sleeping")
    sleep(60)
    # c1, c2 = init_layers()
    # for i in range(1):
    #     forward(a, b, c1, c2)