import mxnet as mx
import logging
import os
import datetime

import sys
sys.path.append('..')
from arg_parsing import process_args
from logger import construct_run_id, configure_root_logger
from data_loaders.cifar10 import Cifar10
from models.preActResnet164_basic_stanford_pytorch_convert import preActResnet164BasicStanfordPytorch
from learners.gluon import GluonLearner


if __name__ == "__main__":
    run_id = construct_run_id(__file__)
    configure_root_logger(run_id)
    logging.info(__file__)

    args = process_args()
    mx.random.seed(args.seed)

    batch_size = 128
    train_data, valid_data = Cifar10(batch_size=batch_size,
                                          data_shape=(3, 32, 32),
                                          padding=2,
                                          padding_value=0,
                                          normalization_type="pixel").return_dataloaders()

    lr_schedule = {0: 0.01, 138: 0.001, 183: 0.0001}

    model = preActResnet164BasicStanfordPytorch(num_classes=10)

    learner = GluonLearner(model, run_id, gpu_idxs=args.gpu_idxs, hybridize=True)
    learner.fit(train_data=train_data,
                 valid_data=valid_data,
                 epochs=300,
                 lr_schedule=lr_schedule,
                 initializer=mx.init.Xavier(rnd_type='uniform', factor_type='out', magnitude=2),
                 optimizer=mx.optimizer.SGD(learning_rate=lr_schedule[0], rescale_grad=1.0/batch_size, momentum=0.9, wd=0.0005),
                 early_stopping_criteria=lambda e: e > 0.94) # DAWNBench CIFAR-10 criteria