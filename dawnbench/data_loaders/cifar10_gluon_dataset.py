import os
import mxnet as mx
import numpy as np


class Cifar10():
    """
    http://data.mxnet.io/mxnet/data/cifar10.zip
    """
    def __init__(self,
                 batch_size,
                 data_shape,
                 padding=None,
                 padding_value=0,
                 normalization_type=None):
        """

        Parameters
        ----------
        batch_size : int
        data_shape
        padding : int
            Number of pixels to pad on each side (top, bottom, left and right)
        padding_value : int
            Value for padded pixels
        normalization_type : str, optional
            Should be either "pixel" or "channel"

        """
        # if normalization_type:
        #     assert normalization_type in ["pixel", "channel"]
        #
        # def transform_aug(data, label, padding=None, padding_value=0):
        #     # transform data
        #     if padding:
        #         data = mx.nd.pad(data, pad_width=(0,0,0,0,padding,padding,padding,padding),
        #                          mode='constant', constant_value=padding_value)
        #     data = data.astype('float32') # convert type
        #     data = data/255 # normalize values
        #     auglist = mx.image.CreateAugmenter(data_shape=(3, 32, 32),
        #                                        resize=0, rand_mirror=True, rand_crop=True,
        #                                        mean=np.array([0.4914, 0.4822, 0.4465]),
        #                                        std=np.array([0.2023, 0.1994, 0.2010]))
        #     for aug in auglist:
        #         data = aug(data)
        #     data = mx.nd.transpose(data, (2, 0, 1))  # channel x width x height
        #     # transform label
        #     label = mx.nd.array([label]).astype('float32')
        #     return data, label
        #
        # def transform_aug_test(data, label, padding=None, padding_value=0):
        #     data = data.astype('float32') # convert type
        #     data = data/255 # normalize values
        #     auglist = mx.image.CreateAugmenter(data_shape=(3, 32, 32),
        #                                        mean=np.array([0.4914, 0.4822, 0.4465]),
        #                                        std=np.array([0.2023, 0.1994, 0.2010]))
        #     for aug in auglist:
        #         data = aug(data)
        #     data = mx.nd.transpose(data, (2, 0, 1))  # channel x width x height
        #     # transform label
        #     label = mx.nd.array([label]).astype('float32')
        #     return data, label

        self.train_dataset = mx.gluon.data.vision.CIFAR10(train=True, transform=lambda e,l: (e.astype('float32')/255, l))
        self.test_dataset = mx.gluon.data.vision.CIFAR10(train=False, transform=lambda e,l: (e.astype('float32')/255, l))
        self.train_dataloader = mx.gluon.data.DataLoader(self.train_dataset, batch_size=batch_size)
        self.test_dataloader = mx.gluon.data.DataLoader(self.test_dataset, batch_size=batch_size)


    def return_dataloaders(self):
        return self.train_dataloader, self.test_dataloader