import sys
sys.path.append("..")


import mxnet as mx
from matplotlib import pyplot as plt
from data_loaders.cifar10_gluon_dataset import Cifar10
from data_loaders.mixup import MixupLoader


def inspect_dataloader(dataloader, n_samples=5):
    """
    :param stage: str, 'train' or 'test'
    :param n_samples: number of images to sample
    :param idx: int, index of image to slice out of batch
    :return: None, just plots
    """
    counter = 0
    try:
        for data_batch, label_batch in dataloader:
            batch_size = data_batch.shape[0]
            for sample_idx in range(batch_size):
                sample = data_batch[sample_idx]
                sample = sample.transpose((1, 2, 0)).asnumpy()
                print("Max value: {}".format(sample.max()))
                print("Min value: {}".format(sample.min()))
                plt.imshow(sample)
                plt.show()
                counter = counter + 1
                if counter >= n_samples:
                    raise StopIteration
    except StopIteration:
        pass


def transform(data, label):
    data = data.transpose(axes=(2, 0, 1))
    data = data.astype('float32')
    data = data / 255
    return data, label


batch_size = 32
train_data_loader, valid_data_loader = Cifar10(batch_size=batch_size,
                                               data_shape=(3, 32, 32),
                                               padding=0,
                                               padding_value=0,
                                               normalization_type="channel").return_dataloaders()
inspect_dataloader(train_data_loader)
    