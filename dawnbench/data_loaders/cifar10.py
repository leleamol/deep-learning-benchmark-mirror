import os
import logging
import mxnet as mx
# from multiprocessing import cpu_count

from .converters import DataIterLoader


class Cifar10():
    """
    http://data.mxnet.io/mxnet/data/cifar10.zip
    """
    def __init__(self,
                 batch_size,
                 data_shape,
                 padding=0,
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

        self.download()
        self.prepare_iters(batch_size, data_shape, normalization_type, padding, padding_value)

    def download(self):
        parent_path = os.path.dirname(os.path.realpath(__file__))
        # two directories higher than current file
        self.data_path = data_path = os.path.abspath(os.path.join(parent_path, "..", "..", "data"))
        cifar_path = os.path.join(data_path, "cifar")

        if not os.path.isdir(data_path):
            os.system("mkdir " + data_path)
        if (not os.path.exists(os.path.join(cifar_path, 'train.rec'))) or \
                (not os.path.exists(os.path.join(cifar_path, 'test.rec'))) or \
                (not os.path.exists(os.path.join(cifar_path, 'train.lst'))) or \
                (not os.path.exists(os.path.join(cifar_path, 'test.lst'))):
            logging.info("Couldn't find CIFAR10 data, downloading...")
            os.system("wget -q http://data.mxnet.io/mxnet/data/cifar10.zip -P " + data_path)
            logging.info("Download complete.")
            os.system("unzip -u " + os.path.join(data_path, "cifar10.zip") + " -d " + data_path)

    def prepare_iters(self, batch_size, data_shape, normalization_type, padding, padding_value):

        shared_args = {'data_shape': data_shape,
                       'batch_size': batch_size}
                       # 'preprocess_threads': cpu_count()}

        if normalization_type == "channel":
            shared_args.update({
                'mean_r': 0.4914*255,
                'mean_g': 0.4822*255,
                'mean_b': 0.4465*255,
                'std_r': 0.2023*255,
                'std_g': 0.1994*255,
                'std_b': 0.2010*255
            })
        elif normalization_type == "pixel":
            shared_args.update({
                'mean_img': os.path.join(self.data_path, "cifar/mean.bin")
            })

        self.train_iter_args = shared_args.copy()
        self.train_iter_args.update({
            'path_imgrec': os.path.join(self.data_path, "cifar/train.rec"),
            'shuffle': True,
            'rand_crop': True,
            'rand_mirror': True,
            'pad': padding,
            'fill_value': padding_value
        })
        self.train_iter = mx.io.ImageRecordIter(**self.train_iter_args)

        self.test_iter_args = shared_args.copy()
        self.test_iter_args.update({'path_imgrec': os.path.join(self.data_path, "cifar/test.rec")})
        self.test_iter = mx.io.ImageRecordIter(**self.test_iter_args)

    def return_dataloaders(self):
        return DataIterLoader(self.train_iter), DataIterLoader(self.test_iter)

    def return_dataiters(self):
        return self.train_iter, self.test_iter