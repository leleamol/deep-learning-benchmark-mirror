import mxnet as mx
import logging
import os
import sys
import numpy as np
import math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arg_parsing import process_args
from logger import construct_run_id, configure_root_logger
from data_loaders.cifar10 import Cifar10
from models.resnet164_basic import resnet164Basic
from learners.gluon_iter import GluonLearner


if __name__ == "__main__":
    run_id = construct_run_id(__file__)
    configure_root_logger(run_id)
    logging.info(__file__)

    args = process_args()
    mx.random.seed(args.seed)

    batch_size = 128
    train_data, valid_data = Cifar10(batch_size=batch_size,
                                          data_shape=(3, 32, 32),
                                          padding=4,
                                          padding_value=0,
                                          normalization_type="channel").return_dataloaders()

    lr_schedule = {0: 0.01, 5: 0.1, 95: 0.01, 140: 0.001}

    model = resnet164Basic(num_classes=10)
    learner = GluonLearner(model, run_id, gpu_idxs=args.gpu_idxs, hybridize=True, tensorboard_logging=True)

    # learner.find_lr(train_data)

    class CyclicalScheduler():
        def __init__(self, stepsize, base_lr, max_lr):
            self.stepsize = stepsize
            self.base_lr = base_lr
            self.max_lr = max_lr

        def get(self, iteration):
            cycle = math.floor(1 + iteration / (2 * self.stepsize))
            x = abs(iteration / self.stepsize - 2 * cycle + 1)
            lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x))
            return lr

        def get_schedule(self, iterations):
            iter_range = range(iterations)
            return {i:self.get(i) for i in iter_range}

    epochs = 22
    samples = 50000
    iters_per_epoch = int(samples/batch_size)
    lr_schedule = CyclicalScheduler(stepsize=iters_per_epoch*5, base_lr=0.001, max_lr=0.1).get_schedule(epochs * iters_per_epoch)

    learner.fit(train_data=train_data,
                 valid_data=valid_data,
                 epochs=epochs,
                 lr_schedule=lr_schedule,
                 initializer=mx.init.Xavier(rnd_type='gaussian', factor_type='out', magnitude=2),
                 optimizer=mx.optimizer.SGD(learning_rate=lr_schedule[0], rescale_grad=1.0/batch_size, momentum=0.9, wd=0.0005),
                 early_stopping_criteria=lambda e: e >= 0.94) # DAWNBench CIFAR-10 criteria