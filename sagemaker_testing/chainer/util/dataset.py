# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
import os
import random
from collections import OrderedDict

import chainer
import numpy as np


class ImageNetDirectoryDataset(chainer.dataset.DatasetMixin):
    def __init__(self, root):
        self._root = root
        self._labels = self._read_label_list(root)
        self._paths = self._read_paths(root, self._labels)
        self._base = chainer.datasets.ImageDataset(self._paths, root)

    def _read_paths(self, root, labels):
        paths = []

        for label in labels.keys():
            files = os.listdir(os.path.join(root, label))
            for f in files:
                if self._is_image_file(os.path.join(root, label, f)):
                    paths.append(os.path.join(label, f))

        return paths

    def _is_image_file(self, path):
        img_extensions = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp'}
        return os.path.isfile(path) and os.path.splitext(path)[1].lower() in img_extensions

    def _read_label_list(self, root):
        labels = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        labels.sort()
        label_dict = OrderedDict()
        for label in labels:
            label_dict[label] = len(label_dict)
        return label_dict

    def _get_label(self, index):
        path = self._paths[index]
        label = os.path.basename(os.path.dirname(path))
        return self._labels[label]

    def get_example(self, index):
        label = self._get_label(index)
        image = self._base[index]
        return image, label

    def __len__(self):
        return len(self._base)


class NormalizedImageNetDataset(chainer.dataset.DatasetMixin):
    def __init__(self, base, mean):
        self._base = base
        self._mean = mean

    def get_example(self, index):
        image, label = self._base[index]
        image -= self._mean  # TODO check for shape compatibility on 1st use
        image *= (1.0 / 255.0)
        return image, label

    def __len__(self):
        return len(self._base)


class CroppedImageDataset(chainer.dataset.DatasetMixin):
    def __init__(self, base, crop_size, random_crop=True):
        self._base = base
        self._crop_size = crop_size
        self._random_crop = random_crop

    def get_example(self, index):
        image, label = self._base[index]
        _, h, w = image.shape

        if self._random_crop:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - self._crop_size - 1)
            left = random.randint(0, w - self._crop_size - 1)
            if random.randint(0, 1):
                image = image[:, :, ::-1]
        else:
            # Crop the center
            top = (h - self._crop_size) // 2
            left = (w - self._crop_size) // 2
        bottom = top + self._crop_size
        right = left + self._crop_size

        image = image[:, top:bottom, left:right]
        return image, label

    def __len__(self):
        return len(self._base)


class SyntheticDataset(chainer.dataset.DatasetMixin):
    def __init__(self, shape, num_classes):
        self._shape = shape
        self._x = (np.random.random(shape) * 255.0).astype(np.float32)
        self._y = np.random.randint(0, num_classes, (shape[0],))

    def get_example(self, i):
        return self._x[i], self._y[i]

    def __len__(self):
        return self._shape[0]
