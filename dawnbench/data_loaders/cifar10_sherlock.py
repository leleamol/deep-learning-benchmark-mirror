import os
import logging
import mxnet as mx

from mxnet import image
from mxnet import nd
from mxnet.gluon.data.vision import ImageFolderDataset
from mxnet.gluon.data import DataLoader
import numpy as np
import shutil


class Cifar10():
    """
    Data from Kaggle CIFAR-10 competition at https://www.kaggle.com/c/cifar-10/data.
    Both training and validation sets are taken from `train.7z`
    And `test.7z` is not used.
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
        if normalization_type:
            assert normalization_type in ["pixel", "channel"]

        parent_path = os.path.dirname(os.path.realpath(__file__))
        # two directories higher than current file
        self.data_path = data_path = os.path.abspath(os.path.join(parent_path, "..", "..", "data","cifar10_sherlock"))

        # self.download()
        self.prepare_iters(batch_size, data_shape, normalization_type, padding, padding_value)

    def download(self):
        raise NotImplementedError

    def preprocess(self):
        with open('./data/trainLabels.csv', 'r') as f:
            lines = f.readlines()[1:]
            tokens = [i.rstrip().split(',') for i in lines]
            idx_label = dict((int(idx), label) for idx, label in tokens)
        labels = set(idx_label.values())

        num_train = len(os.listdir('./data/train/'))

        num_train_tuning = int(num_train * (1 - 0.1))

        num_train_tuning_per_label = num_train_tuning // len(labels)

        label_count = dict()

        def mkdir_if_not_exist(path):
            if not os.path.exists(os.path.join(*path)):
                os.makedirs(os.path.join(*path))

        for train_file in os.listdir('./data/train/'):
            idx = int(train_file.split('.')[0])
            label = idx_label[idx]
            mkdir_if_not_exist(['./data', 'train_valid', label])
            shutil.copy(os.path.join('./data/train/', train_file),
                        os.path.join('./data/train_valid', label))
            if label not in label_count or label_count[label] < num_train_tuning_per_label:
                mkdir_if_not_exist(['./data/train_data', label])
                shutil.copy(os.path.join('./data/train', train_file),
                            os.path.join('./data/train_data', label))
                label_count[label] = label_count.get(label, 0) + 1
            else:
                mkdir_if_not_exist(['./data/valid_data', label])
                shutil.copy(os.path.join('./data/train/', train_file),
                            os.path.join('./data/valid_data', label))


    def transform_train(self, data, label):
        im = data.asnumpy()
        im = np.pad(im, ((4, 4), (4, 4), (0, 0)), mode='constant', constant_values=0)
        im = nd.array(im, dtype='float32') / 255
        auglist = image.CreateAugmenter(data_shape=(3, 32, 32), resize=0, rand_mirror=True,
                                        rand_crop=True,
                                        mean=np.array([0.4914, 0.4822, 0.4465]),
                                        std=np.array([0.2023, 0.1994, 0.2010]))
        # different from channel means on fast,ai course [0.4914, 0.48216, 0.44653],
        # [0.24703, 0.24349, 0.26159]
        for aug in auglist:
            im = aug(im)
        im = nd.transpose(im, (2, 0, 1))  # channel x width x height
        return im, nd.array([label]).astype('float32')


    def transform_test(self, data, label):
        im = data.astype('float32') / 255
        auglist = image.CreateAugmenter(data_shape=(3, 32, 32), mean=np.array([0.4914, 0.4822, 0.4465]),
                                        std=np.array([0.2023, 0.1994, 0.2010]))
        for aug in auglist:
            im = aug(im)
        im = nd.transpose(im, (2, 0, 1))
        return im, nd.array([label]).astype('float32')


    def prepare_iters(self, batch_size, data_shape, normalization_type, padding, padding_value):
        train_ds = ImageFolderDataset(os.path.join(self.data_path, 'train_data'), transform=self.transform_train)
        test_ds = ImageFolderDataset(os.path.join(self.data_path, 'valid_data'), transform=self.transform_test)

        self.train_data = DataLoader(train_ds, batch_size=64, shuffle=True, last_batch='keep')
        self.test_data = DataLoader(test_ds, batch_size=64, shuffle=True, last_batch='keep')


    def return_dataloaders(self):
        return self.train_data, self.test_data