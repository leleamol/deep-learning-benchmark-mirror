import mxnet as mx
from mxboard import SummaryWriter
import os

import sys
sys.path.append('..')
from data_loaders.cifar10 import Cifar10

current_folder = os.path.dirname(os.path.realpath(__file__))
tensorboard_folder = os.path.realpath(os.path.join(current_folder, "..", "..", "logs", "tensorboard"))
summary_filepath = os.path.join(tensorboard_folder, "mxnet_data_inspect")
print(summary_filepath)

mx.random.seed(42)

batch_size = 128
train_data, valid_data = Cifar10(batch_size=batch_size,
                                 data_shape=(3, 32, 32),
                                 padding=4,
                                 padding_value=0,
                                 normalization_type="channel").return_dataloaders()

with SummaryWriter(logdir=summary_filepath) as writer:
    for batch in train_data:
        batch_data = batch[0]
        break
    writer.add_image(tag="batch", image=batch_data, global_step=1)
    writer.add_image(tag="sample", image=batch_data[0], global_step=1)
    writer.add_histogram(tag='batch_values', values=batch_data, global_step=2, bins=100)
    writer.add_histogram(tag='batch_values_red', values=batch_data[:,0], global_step=1, bins=100)
    writer.add_histogram(tag='batch_values_green', values=batch_data[:,1], global_step=1, bins=100)
    writer.add_histogram(tag='batch_values_blue', values=batch_data[:,2], global_step=1, bins=100)