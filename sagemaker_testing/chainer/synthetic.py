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
import argparse
import json
import logging
import os

import chainer
import chainer.links as L
import numpy as np
from chainer.training import extensions

import models
from util import dataset
from util import metrics
from util import report
from util import trigger

MODELS = {
    'resnet50': ((3, 224, 224), 1000, lambda: models.ResNet50()),
    'vgg': ((3, 32, 32), 10, lambda: L.Classifier(models.vgg.VGG(10)))
}

logger = logging.getLogger(__name__)


def train(args):
    try:
        sm_env = json.loads(os.environ['SM_TRAINING_ENV'])
        hosts = sm_env['hosts']
        num_gpus = sm_env['num_gpus']
        output_data_dir = sm_env['output_data_dir']

        model_name = args.model_name
        samples = args.samples
        batch_size = args.batch_size
        epochs = args.epochs
        learning_rate = args.learning_rate
        max_time_seconds = args.max_time
        devices = range(num_gpus) if num_gpus > 0 else [-1]

        model_shape, num_classes, model_fn = MODELS[model_name]
        dataset_shape = (samples,) + model_shape
        model = model_fn()

        total_dataset_bytes = np.prod(dataset_shape)
        train = dataset.SyntheticDataset(dataset_shape, num_classes)

        optimizer = chainer.optimizers.MomentumSGD(learning_rate)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

        if len(devices) > 1:
            train_iters = [chainer.iterators.SerialIterator(d, batch_size) for d in
                           chainer.datasets.split_dataset_n_random(train, len(devices))]
            updater = chainer.training.updaters.MultiprocessParallelUpdater(train_iters, optimizer,
                                                                            devices=devices)
        else:
            train_iter = chainer.iterators.SerialIterator(train, batch_size)
            updater = chainer.training.updaters.StandardUpdater(train_iter, optimizer,
                                                                device=devices[0])

        triggers = [
            trigger.TimeLimitTrigger(max_time_seconds),
            chainer.training.triggers.IntervalTrigger(epochs, 'epoch'),
        ]
        composite_trigger = trigger.CompositeTrigger(triggers, max_trigger=(epochs, 'epoch'))

        trainer = chainer.training.Trainer(
            updater, stop_trigger=composite_trigger, out=output_data_dir)
        trainer.extend(extensions.ProgressBar(update_interval=1))

        metrics_agent = metrics.Metrics({
            'framework': 'chainer',
            'framework_version': '4.0.0',
            'instance_count': str(len(hosts)),
            'batch_size': str(batch_size),
            'instance_type': args.instance_type,
            'model': model_name
        })
        trainer.extend(report.MetricsReport(parallelism=len(devices), agent=metrics_agent,
                                            dataset_length=len(train),
                                            dataset_mb=total_dataset_bytes))
        metrics_agent.start()
        trainer.run()

        actual_epochs = trainer.updater.epoch_detail
        job_samples_sec = actual_epochs * len(train) / trainer.elapsed_time
        job_mb_sec = actual_epochs * total_dataset_bytes / trainer.elapsed_time / 1024 ** 2
        metrics_agent.update('job_samples_sec', job_samples_sec)
        metrics_agent.update('job_mb_sec', job_mb_sec)
        metrics_agent.stop()

        return model

    except Exception as e:
        logger.exception('exception during training!')
        raise e


# -----------------------------
# for local testing outside SM
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='chainer benchmark script')
    parser.set_defaults(func=lambda x: parser.print_usage())
    parser.add_argument('--model-name', type=str, default='resnet50')
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--max-time', type=int, default=3600)
    parser.add_argument('--log-level', type=str, default='info')
    parser.add_argument('--instance-type', type=str, default='unknown')
    parser.add_argument('--gpus', type=int, default=-1)

    return parser.parse_args()


def main():
    args = parse_args()
    log_format = '%(asctime)s %(levelname)s %(name)s: %(message)s'
    log_level = logging.getLevelName(args.log_level.upper())
    logging.basicConfig(format=log_format, level=log_level)
    logging.getLogger('botocore').setLevel(logging.WARN)

    train(args)


if __name__ == '__main__':
    main()
