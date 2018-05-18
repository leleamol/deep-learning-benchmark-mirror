import mxnet as mx
import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arg_parsing import process_args
from logger import construct_run_id, configure_root_logger
from data_loaders.cifar10 import Cifar10
from models.resnet164_basic import resnet164Basic
from learners.module import ModuleLearner


if __name__ == "__main__":
    run_id = construct_run_id(__file__)
    configure_root_logger(run_id)
    logging.info(__file__)

    args = process_args()
    mx.random.seed(args.seed)

    _, test_data = Cifar10(batch_size=1, data_shape=(3, 32, 32),
                           normalization_type="channel").return_dataiters()

    # download model symbol and params (if doesn't already exist)
    for filename in ["resnet164_basic_module-0000.params", "resnet164_basic_module-symbol.json"]:
        folder = os.path.realpath(os.path.join(os.path.realpath(__file__), "../logs/checkpoints/"))
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            os.system("aws s3 cp s3://benchmark-ai-models/{} {}".format(filename, folder))

    model = resnet164Basic(num_classes=10)
    learner = ModuleLearner(model, run_id, gpu_idxs=args.gpu_idxs)
    learner.load(prefix="resnet164_basic_module", data_iter=test_data)
    learner.predict(test_data=test_data, log_frequency=100)