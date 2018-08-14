import mxnet as mx
import logging
import os
import datetime

import sys
sys.path.append('..')
from arg_parsing import process_args
from logger import construct_run_id, configure_root_logger
from data_loaders.cifar10 import Cifar10
from data_loaders.mixup import OnehotLoader, MixupClassLoader
from learners.gluon import GluonLearner


if __name__ == "__main__":
    run_id = construct_run_id(__file__)
    configure_root_logger(run_id)
    logging.info(__file__)

    args = process_args()
    mx.random.seed(args.seed)

    batch_size = 512
    train_data, valid_data = Cifar10(batch_size=batch_size,
                                          data_shape=(3, 32, 32),
                                          padding=4,
                                          padding_value=0,
                                          normalization_type="channel").return_dataloaders()
    train_data = MixupClassLoader(train_data, num_classes=10, alpha=10)
    # valid_data = OnehotLoader(valid_data, num_classes=10)

    from mxboard import SummaryWriter

    current_folder = os.path.dirname(os.path.realpath(__file__))
    tensorboard_folder = os.path.join(current_folder, "..", "logs", "tensorboard")
    summary_filepath = os.path.join(tensorboard_folder, "testthis7")

    writer = SummaryWriter(logdir=summary_filepath)
    writer.add_scalar(tag='learning_rate', value=123, global_step=0 + 1)
    for idx, batch in enumerate(train_data):
        writer.add_image(tag="batch", image=batch[0], global_step=idx + 1)
        if idx>2:
            break
    writer.flush()
    writer.close()

    # learning_rate = 0.01
    # lr_schedule = {0: 0.1, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1,
    #                80: 0.1,
    #                120: 0.01, 160: 0.003, 200: 0.001}
    # lr_schedule = {k: v*learning_rate for (k, v) in lr_schedule.items()}
    #
    # model = mx.gluon.model_zoo.vision.get_model(name="resnet50_v2", classes=10)
    #
    # learner = GluonLearner(model, run_id, gpu_idxs=args.gpu_idxs, hybridize=True, tensorboard_logging=True)
    # learner.fit(train_data=train_data,
    #              valid_data=valid_data,
    #              epochs=200,
    #              lr_schedule=lr_schedule,
    #              initializer=mx.init.Xavier(rnd_type='uniform', factor_type='out', magnitude=2),
    #              optimizer=mx.optimizer.SGD(learning_rate=lr_schedule[0], rescale_grad=1.0/batch_size, momentum=0.9, wd=0.0005),
    #              early_stopping_criteria=lambda e: e > 0.94) # DAWNBench CIFAR-10 criteria)