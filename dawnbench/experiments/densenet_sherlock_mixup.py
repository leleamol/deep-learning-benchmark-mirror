import mxnet as mx
import logging
import os
import datetime

import sys
sys.path.append('..')
from arg_parsing import process_args
from logger import construct_run_id, configure_root_logger
from data_loaders.cifar10 import Cifar10
from data_loaders.mixup import OnehotLoader, MixupLoader
from models.densenet_sherlock import DenseNet
from learners.gluon import GluonLearner


if __name__ == "__main__":
    run_id = construct_run_id(__file__)
    configure_root_logger(run_id)
    logging.info(__file__)

    args = process_args()
    mx.random.seed(args.seed)

    batch_size = 256
    train_data, valid_data = Cifar10(batch_size=batch_size,
                                          data_shape=(3, 32, 32),
                                          padding=4,
                                          padding_value=0,
                                          normalization_type="pixel").return_dataloaders()
    train_data = MixupLoader(train_data, num_classes=10, alpha=1)
    valid_data = OnehotLoader(valid_data, num_classes=10)

    lr_schedule = {0: 0.1,
                   149: 0.01,
                   224: 0.001}

    # DenseNet-BC
    model = DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=10)

    learner = GluonLearner(model, run_id, gpu_idxs=args.gpu_idxs, hybridize=True)
    learner.fit(train_data=train_data,
                 valid_data=valid_data,
                 epochs=300,
                 lr_schedule=lr_schedule,
                 initializer=mx.init.Xavier(rnd_type='uniform', factor_type='out', magnitude=2),
                 optimizer=mx.optimizer.SGD(learning_rate=lr_schedule[0], rescale_grad=1.0/batch_size, momentum=0.9, wd=0.0001),
                 early_stopping_criteria=lambda e: e > 0.94, # DAWNBench CIFAR-10 criteria
                 sparse_label=False)