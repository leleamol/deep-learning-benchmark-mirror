import os

def getImagenetData(dataset):

    if dataset == 'imagenet':
        os.system('mkdir -p data')
        if not os.path.exists('./data/imagenet1k-train.rec'):
            os.system("aws s3 cp  s3://imagenet-rec/imagenet1k-train.rec data/")
        if not os.path.exists('./data/imagenet1k-val.rec'):
            os.system("aws s3 cp  s3://imagenet-rec/imagenet1k-val.rec data/")

    elif dataset == 'imagenet-256px-q90':
        if not os.path.exists(os.path.expanduser('~/data/imagenet1k-train.rec')):
            os.system("wget -q https://s3.amazonaws.com/ragab-datasets/imagenet2012/imagenet1k-train.rec -P ~/data/")
        if not os.path.exists(os.path.expanduser('~/data/imagenet1k-val.rec')):
            os.system("wget -q https://s3.amazonaws.com/ragab-datasets/imagenet2012/imagenet1k-val.rec -P ~/data/")
    elif dataset == 'imagenet-480px-q95':
        if not os.path.exists(os.path.expanduser('~/data/train-480px-q95.rec')):
            os.system("wget -q https://s3.amazonaws.com/aws-ml-platform-datasets/imagenet/480px-q95/train-480px-q95.rec -P ~/data/")
        if not os.path.exists(os.path.expanduser('~/data/val-256px-q95.rec')):
            os.system("wget -q https://s3.amazonaws.com/aws-ml-platform-datasets/imagenet/256px-q95/val-256px-q95.rec -P ~/data/")
    elif dataset == 'imagenet-480px-256px-q95':
        if not os.path.exists(os.path.expanduser('~/data/train-480px-q95.rec')):
            os.system("wget -q https://s3.amazonaws.com/aws-ml-platform-datasets/imagenet/480px-q95/train-480px-q95.rec -P ~/data/")
        if not os.path.exists(os.path.expanduser('~/data/val-256px-q95.rec')):
            os.system("wget -q https://s3.amazonaws.com/aws-ml-platform-datasets/imagenet/256px-q95/val-256px-q95.rec -P ~/data/")
        if not os.path.exists(os.path.expanduser('~/data/train-256px-q95.rec')):
            os.system("wget -q https://s3.amazonaws.com/aws-ml-platform-datase	ts/imagenet/256px-q95/train-256px-q95.rec -P ~/data/")
        
    else:
        raise ValueError('Unknown dataset')
