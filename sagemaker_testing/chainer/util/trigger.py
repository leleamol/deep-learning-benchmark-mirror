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

from chainer.training import util

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CompositeTrigger(object):
    """Composite Trigger

    Composite Trigger allows multiple triggers to be combined for use in
    settings where only one Trigger can be specified. Ideal for applying
    multiple stopping conditions to a Trainer.

    The Trigger objects will be evaluated in the order they were passed
    into `__init__`, and evaluation will stop after the first `True`.

    Args:
        triggers (Iterable of Trigger): one or more Trigger objects.
        max_trigger ((int, string)): tuple of period, unit representing
            upper bound on training length

    """

    def __init__(self, triggers, max_trigger=(100, 'epoch')):
        self._triggers = list(triggers)
        self._max_trigger = util.get_trigger(max_trigger)

        # find minimum training length
        lengths = [t.get_training_length() for t in triggers if hasattr(t, 'get_training_length')]
        lengths = sorted(lengths, key=lambda x: (1 if x[1] == 'epoch' else 0, x[0]), reverse=True)
        self._training_length = lengths[0] or max_trigger
        logger.info('training length: {}'.format(self._training_length))

    def __call__(self, trainer):
        """Decides whether the extension should be called on this iteration.

          Args:
              trainer (Trainer): Trainer object that this trigger is associated
                  with. The updater associated with this trainer is used to
                  determine if the trigger should fire.

          Returns:
              bool: True if the corresponding extension should be invoked in this
              iteration.
          """
        return any(t(trainer) for t in self._triggers)

    def get_training_length(self):
        return self._training_length


class TimeLimitTrigger(object):
    def __init__(self, limit_seconds):
        self._limit = limit_seconds

    def __call__(self, trainer):
        triggered = self._limit <= trainer.elapsed_time
        if triggered:
            logger.info('time limit reached (limit: {}, elapsed: {})'.format(self._limit,
                                                                             trainer.elapsed_time))
        return triggered


class ValidationAccuracyTrigger(object):
    def __init__(self, limit=0.93):
        self._limit = limit

    def __call__(self, trainer):
        observation = trainer.observation

        if 'validation/main/accuracy' in observation:
            accuracy = float(observation['validation/main/accuracy'].get())
            triggered = accuracy >= self._limit
            if triggered:
                logger.info(
                    'accuracy threshold reached (accuracy: {}, epoch: {}, iteration: {})'.format(
                        accuracy, trainer.updater.epoch, trainer.updater.iteration))
            return triggered
        return False
