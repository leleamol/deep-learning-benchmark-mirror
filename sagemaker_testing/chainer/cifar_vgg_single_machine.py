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

from __future__ import print_function, absolute_import

import logging
import os

import chainer
import chainer.links as L
import numpy as np
from chainer import training
from chainer.training import extensions
from util import report
from util import trigger
from models import vgg
from util import metrics

TOTAL_DATASET_MB = 134.73


def train(hyperparameters, num_gpus, output_data_dir, channel_input_dirs, hosts):
    train_data = np.load(os.path.join(channel_input_dirs['train'], 'train.npz'))['data']
    train_labels = np.load(os.path.join(channel_input_dirs['train'], 'train.npz'))['labels']

    test_data = np.load(os.path.join(channel_input_dirs['test'], 'test.npz'))['data']
    test_labels = np.load(os.path.join(channel_input_dirs['test'], 'test.npz'))['labels']

    train = chainer.datasets.TupleDataset(train_data, train_labels)
    test = chainer.datasets.TupleDataset(test_data, test_labels)

    # retrieve the hyperparameters we set in notebook (with some defaults)
    batch_size = hyperparameters.get('batch_size', 64)
    epochs = hyperparameters.get('epochs', 300)
    learning_rate = hyperparameters.get('learning_rate', 0.05)
    num_loaders = hyperparameters.get('num_loaders', 4)

    # Set up a neural network to train.
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model = L.Classifier(vgg.VGG(10))

    optimizer = chainer.optimizers.MomentumSGD(learning_rate)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

    # Set up a trainer
    # device = 0 if num_gpus > 0 else -1  # -1 indicates CPU, 0 indicates first GPU device.
    devices = range(num_gpus) if num_gpus > 0 else [-1]
    if num_gpus > 1:

        train_iters = [
            chainer.iterators.MultiprocessIterator(i, batch_size, n_processes=num_loaders) \
            for i in chainer.datasets.split_dataset_n_random(train, len(devices))]
        test_iter = chainer.iterators.MultiprocessIterator(test, batch_size, repeat=False,
                                                           n_processes=num_loaders)
        updater = training.updaters.MultiprocessParallelUpdater(train_iters, optimizer,
                                                                devices=range(num_loaders))
    else:
        train_iter = chainer.iterators.MultiprocessIterator(train, batch_size)
        test_iter = chainer.iterators.MultiprocessIterator(test, batch_size, repeat=False)
        updater = training.updater.StandardUpdater(train_iter, optimizer, device=devices[0])

    triggers = [
        trigger.ValidationAccuracyTrigger(limit=0.93),
        trigger.TimeLimitTrigger(12 * 60 * 60),
        chainer.training.triggers.IntervalTrigger(epochs, 'epoch'),
    ]
    composite_trigger = trigger.CompositeTrigger(triggers)

    trainer = training.Trainer(updater, stop_trigger=composite_trigger, out=output_data_dir)
    trainer.extend(extensions.Evaluator(test_iter, model, device=devices[0]))
    trainer.extend(extensions.ExponentialShift('lr', 0.5), trigger=(25, 'epoch'))
    trainer.extend(extensions.ProgressBar(update_interval=5))

    metrics_agent = metrics.Metrics({
        'framework': 'chainer',
        'framework_version': '4.0.0',
        'instance_count': str(len(hosts)),
        'batch_size': str(batch_size),
        'instance_type': hyperparameters.get('instance_type', 'unknown')
    })
    trainer.extend(report.MetricsReport(parallelism=len(devices), agent=metrics_agent,
                                        dataset_length=len(train),
                                        dataset_mb=TOTAL_DATASET_MB))

    metrics_agent.start()
    trainer.run()
    metrics_agent.update('job_samples_sec', trainer.updater.epoch_detail * len(
        train) / trainer.elapsed_time)
    metrics_agent.update('job_mb_sec',
                         trainer.updater.epoch_detail * TOTAL_DATASET_MB / trainer.elapsed_time)
    metrics_agent.stop()

    return model


# for local testing outside SageMaker

def prepare_data():
    os.makedirs('data/train', exist_ok=True)
    os.makedirs('data/test', exist_ok=True)

    train, test = chainer.datasets.get_cifar10()
    train_data = [element[0] for element in train]
    train_labels = [element[1] for element in train]

    test_data = [element[0] for element in test]
    test_labels = [element[1] for element in test]
    np.savez('data/train/train.npz', data=train_data, labels=train_labels)
    np.savez('data/test/test.npz', data=test_data, labels=test_labels)


def main():
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('botocore').setLevel(logging.WARN)

    prepare_data()

    channel_input_dirs = {
        'train': os.path.abspath('data/train'),
        'test': os.path.abspath('data/test')
    }

    hp = {}
    train(hp, 0, 'output', channel_input_dirs, ['localhost'])


if __name__ == '__main__':
    main()
