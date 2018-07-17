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
import time

import chainer


class MetricsReport(chainer.training.extension.Extension):
    def __init__(self, agent, parallelism, dataset_length, dataset_mb):
        self._initialized = False
        self.parallelism = parallelism
        self.dataset_mb = dataset_mb
        self.dataset_length = dataset_length
        self.job_start = None
        self.epoch_start = None
        self.batch_start = None
        self.batch_size = None
        self.agent = agent

    def __call__(self, trainer):

        now = time.time()

        # init times
        if not self._initialized:
            self.job_start = now - trainer.elapsed_time
            self.epoch_start = self.job_start
            self.batch_start = self.job_start
            self.batch_size = trainer.updater.get_iterator('main').batch_size
            self._initialized = True

        observation = trainer.observation

        if 'main/accuracy' in observation:
            self.agent.update('training_accuracy_pct',
                              float(observation['main/accuracy'].data) * 100.0)

        if 'validation/main/accuracy' in observation:
            self.agent.update('validation_accuracy_pct',
                              self._validation_accuracy(observation) * 100.0)

        self.agent.update('batch_samples_sec',
                          (self.batch_size * self.parallelism) / (now - self.batch_start))
        self.batch_start = now

        if trainer.updater.is_new_epoch:
            self.agent.update('epoch_samples_sec', self.dataset_length / (now - self.epoch_start))
            self.agent.update('epoch_mb_sec', self.dataset_mb / (now - self.epoch_start))
            self.epoch_start = now

    def _validation_accuracy(self, observation):
        acc = observation['validation/main/accuracy']

        if hasattr(acc, 'get'):
            acc = acc.get()
        return float(acc)
