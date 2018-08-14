import mxnet as mx
import numpy as np

class OnehotLoader():
    def __init__(self, data_loader, num_classes):
        """

        Parameters
        ----------
        data_loader
        alpha: float
            Used as Beta distribution parameter for sampling lambda (the mixup ratio for combining samples).
        """
        self.data_loader = data_loader
        self.num_classes = num_classes

    def __iter__(self):
        self.open_iter = self.data_loader.__iter__()
        return self

    def __next__(self):
        data, label = self.open_iter.__next__()
        if len(label.shape) == 1:
            # must one hot encode label, otherwise label combination will be meaningless
            label = mx.nd.one_hot(label, depth=self.num_classes)
        return data, label

    def next(self):
        return self.__next__() # for Python 2


# class MixupClassLoader():
#     def __init__(self, data_loader, num_classes, alpha=1):
#         """
#         Applies mixup to samples returned by the provided DataLoader.
#         Shuffle is performed within batch, not the whole dataset.
#         See https://arxiv.org/abs/1710.09412
#         Parameters
#         ----------
#         data_loader: DataLoader
#             Label is expected to denote a class, and not be one hot encoded.
#         alpha: float
#             Used as Beta distribution parameter for sampling lambda (the mixup ratio for combining samples).
#         """
#         self.data_loader = data_loader
#         self.num_classes = num_classes
#         self.alpha = alpha
#
#     def __iter__(self):
#         self.open_iter = self.data_loader.__iter__()
#         return self
#
#     def __next__(self):
#         data_orig, label_orig = self.open_iter.__next__()
#         batch_size = data_orig.shape[0]
#         # one hot encode label, otherwise label combination will be meaningless
#         label_orig = mx.nd.one_hot(label_orig, depth=self.num_classes)
#
#         # alternative data and label from a shuffled batch
#         alt_idx = np.arange(batch_size)
#         np.random.shuffle(alt_idx)
#         data_alt, label_alt = (data_orig[alt_idx], label_orig[alt_idx])
#
#         # sample lambdas from beta distribution
#         lambdas = mx.nd.array(np.random.beta(self.alpha, self.alpha, size=batch_size))
#
#         # reshape lambda for data (so broadcasting can be used)
#         d_reshape = [-1 if i == 0 else 1 for i in range(len(data_orig.shape))]
#         d_lambdas = lambdas.reshape(shape=d_reshape)
#         # reshape lambda for label (so broadcasting can be used)
#         l_reshape = [-1 if i == 0 else 1 for i in range(len(label_orig.shape))]
#         l_lambdas = lambdas.reshape(shape=l_reshape)
#
#         # combine samples
#         data = d_lambdas * data_orig + (1-d_lambdas) * data_alt
#         label = l_lambdas * label_orig + (1-l_lambdas) * label_alt # must be one hot encoded float labels
#
#         return data, label
#
#     def next(self):
#         return self.__next__() # for Python 2



class MixupClassLoader():
    def __init__(self, data_loader, num_classes, alpha=1):
        """
        Applies mixup to samples returned by the provided DataLoader.
        Shuffle is performed within batch, not the whole dataset.
        See https://arxiv.org/abs/1710.09412
        Parameters
        ----------
        data_loader: DataLoader
            Label is expected to denote a class, and not be one hot encoded.
        alpha: float
            Used as Beta distribution parameter for sampling lambdas (the mixup ratios for combining samples).
        """
        self.data_loader = data_loader
        self.num_classes = num_classes
        self.alpha = alpha

    def __iter__(self):
        self.open_iter = self.data_loader.__iter__()
        return self

    def __next__(self):
        data_orig, label_orig = self.open_iter.__next__()
        batch_size = data_orig.shape[0]
        # one hot encode label, otherwise label combination will be meaningless
        label_orig = mx.nd.one_hot(label_orig, depth=self.num_classes)

        # shuffle index to create alternative batch of data and labels
        alt_idx = np.random.permutation(np.arange(batch_size))
        data_alt, label_alt = (data_orig[alt_idx], label_orig[alt_idx])

        # sample lambdas from beta distribution
        lambdas = mx.nd.array(np.random.beta(self.alpha, self.alpha, size=batch_size))
        # reshape lambdas for data and label (so broadcasting can be used)
        d_lambdas = lambdas.reshape(shape=[-1 if i == 0 else 1 for i in range(len(data_orig.shape))])
        l_lambdas = lambdas.reshape(shape=[-1 if i == 0 else 1 for i in range(len(label_orig.shape))])

        # combine samples (of batch)
        data = d_lambdas * data_orig + (1-d_lambdas) * data_alt
        label = l_lambdas * label_orig + (1-l_lambdas) * label_alt # must be one hot encoded float labels

        return data, label

    def next(self):
        return self.__next__() # for Python 2

