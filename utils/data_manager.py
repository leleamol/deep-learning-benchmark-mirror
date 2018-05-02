import os

def getImagenetData(dataset):
    if dataset == 'imagenet' or dataset == 'imagenet-256px-q90':
        if not os.path.exists(os.path.expanduser('~/data/imagenet1k-train.rec')):
            os.system("wget -q https://s3.amazonaws.com/ragab-datasets/imagenet2012/imagenet1k-train.rec -P ~/data/")
        if not os.path.exists(os.path.expanduser('~/data/imagenet1k-val.rec')):
            os.system("wget -q https://s3.amazonaws.com/ragab-datasets/imagenet2012/imagenet1k-val.rec -P ~/data/")
    elif dataset == 'imagenet-480px-q95':
        if not os.path.exists(os.path.expanduser('~/data/imagenet_train.rec')):
            os.system("wget -q https://s3.amazonaws.com/aws-ml-platform-datasets/480px-95quality-imagenet/imagenet_train.rec -P ~/data/")
        if not os.path.exists(os.path.expanduser('~/data/val-256px-q95.rec')):
            os.system("wget -q https://s3.amazonaws.com/aws-ml-platform-datasets/256px-95quality-imagenet/val-256px-q95.rec -P ~/data/")
    elif dataset == 'imagenet-480px-256px-q95':
        if not os.path.exists(os.path.expanduser('~/data/imagenet_train.rec')):
            os.system("wget -q https://s3.amazonaws.com/aws-ml-platform-datasets/480px-95quality-imagenet/imagenet_train.rec -P ~/data/")
        if not os.path.exists(os.path.expanduser('~/data/train-256px-q95.rec')):
            os.system("wget -q https://s3.amazonaws.com/aws-ml-platform-datasets/256px-95quality-imagenet/train-256px-q95.rec -P ~/data/")
        if not os.path.exists(os.path.expanduser('~/data/val-256px-q95.rec')):
            os.system("wget -q https://s3.amazonaws.com/aws-ml-platform-datasets/256px-95quality-imagenet/val-256px-q95.rec -P ~/data/")
    else:
        raise ValueError('Unknown dataset')