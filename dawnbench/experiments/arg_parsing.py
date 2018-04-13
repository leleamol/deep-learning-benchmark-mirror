import argparse
import logging


def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR-10 image classification.')
    parser.add_argument('--num-gpus', type=int, default=0, help='Number of GPUs to use for training. Will use first N GPUs.')
    parser.add_argument('--gpu-idxs', type=str, default='', help='Comma separated list of GPU indices to use for training. Overrides `gpus` argument.')
    parser.add_argument('--seed', type=int, default=42, help='Seed to use for randomization.')
    args = parser.parse_args()
    return args


def process_args():
    args = parse_args()
    if args.gpu_idxs:
        args.gpu_idxs = [int(g) for g in args.gpu_idxs.split(',')]
        logging.info("Using {} GPUs.".format(len(args.gpu_idxs)))
    elif args.num_gpus > 0:
        args.gpu_idxs = [g for g in range(args.num_gpus)]
        logging.info("Using {} GPUs.".format(len(args.gpu_idxs)))
    else:
        args.gpu_idxs = None
        logging.info("Using CPU.")
    return args