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

import logging
import os
import sys

import chainer
import numpy as np
from util import dataset
import models
from util import report
from util import trigger
from util import metrics

logger = logging.getLogger(__name__)
TOTAL_DATASET_MB = 38666.53


def train(hyperparameters, num_gpus, output_data_dir, channel_input_dirs, hosts):
    mean_image_path = os.path.join(channel_input_dirs['resources'], 'mean.npy')
    train_path = channel_input_dirs['train']
    validation_path = channel_input_dirs['validation']

    training_batch_size = hyperparameters.get('training_batch_size', 32)
    validation_batch_size = hyperparameters.get('validation_batch_size', 32)
    epochs = hyperparameters.get('epochs', 3)
    loader_job = hyperparameters.get('loader_job', 4)

    # Initialize the model to train
    model = models.ResNet50()

    # Load the datasets and mean file
    mean = np.load(mean_image_path)

    training_dataset = dataset.CroppedImageDataset(
        dataset.NormalizedImageNetDataset(
            dataset.ImageNetDirectoryDataset(root=train_path), mean=mean),
        crop_size=model.insize, random_crop=True)

    validation_dataset = dataset.CroppedImageDataset(
        dataset.NormalizedImageNetDataset(
            dataset.ImageNetDirectoryDataset(root=validation_path), mean=mean),
        crop_size=model.insize, random_crop=False)

    # These iterators load the images with subprocesses running in parallel to
    # the training/validation.
    devices = range(num_gpus)

    train_iters = [
        chainer.iterators.MultiprocessIterator(i,
                                               training_batch_size,
                                               n_processes=loader_job)
        for i in chainer.datasets.split_dataset_n_random(training_dataset, len(devices))]
    val_iter = chainer.iterators.MultiprocessIterator(
        validation_dataset, validation_batch_size, repeat=False, n_processes=loader_job)

    # Set up an optimizer
    optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)

    # Set up a trainer
    updater = chainer.training.updaters.MultiprocessParallelUpdater(
        train_iters, optimizer, devices=devices)

    triggers = [
        trigger.ValidationAccuracyTrigger(limit=0.93),
        trigger.TimeLimitTrigger(12 * 60 * 60),
        chainer.training.triggers.IntervalTrigger(epochs, 'epoch'),
    ]
    composite_trigger = trigger.CompositeTrigger(triggers)

    trainer = chainer.training.Trainer(
        updater, stop_trigger=composite_trigger, out=output_data_dir)

    metrics_agent = metrics.Metrics({
        'framework': 'chainer',
        'framework_version': chainer.__version__,
        'instance_count': str(len(hosts)),
        'batch_size': str(training_batch_size),
        'instance_type': hyperparameters.get('instance_type', 'unknown'),
        'model': 'resnet50'
    })

    trainer.extend(chainer.training.extensions.Evaluator(val_iter, model, device=devices[0]))
    trainer.extend(report.MetricsReport(parallelism=num_gpus, agent=metrics_agent,
                                        dataset_length=len(training_dataset),
                                        dataset_mb=TOTAL_DATASET_MB))
    trainer.extend(chainer.training.extensions.ProgressBar(update_interval=5))

    metrics_agent.start()
    trainer.run()
    metrics_agent.update('job_samples_sec', trainer.updater.epoch_detail * len(
        training_dataset) / trainer.elapsed_time)
    metrics_agent.update('job_mb_sec',
                         trainer.updater.epoch_detail * TOTAL_DATASET_MB / trainer.elapsed_time)
    metrics_agent.stop()


# for local testing outside SageMaker
def main():
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('botocore').setLevel(logging.WARN)
    # logging.getLogger('scripts.common.metrics').setLevel(logging.DEBUG)
    # logging.getLogger('scripts.chainer.trigger').setLevel(logging.DEBUG)
    channel_input_dirs = {
        'train': '../imagenet/train',
        'validation': '../imagenet/val',
        'resources': '../imagenet/resources'
    }

    hp = {'epochs': 80, 'training_batch_size': 128, 'validation_batch_size': 128}
    train(hp, 16, 'output', channel_input_dirs, ['localhost'])


if __name__ == '__main__':
    main()
