import os

def getImagenetData(dataset):
    if dataset == 'imagenet' or dataset == 'imagenet-256px':
        if not os.path.exists(os.path.expanduser('~/data/imagenet1k-train.rec')):
            os.system("wget -q https://s3.amazonaws.com/ragab-datasets/imagenet2012/imagenet1k-train.rec -P ~/data/")
        if not os.path.exists(os.path.expanduser('~/data/imagenet1k-val.rec')):
            os.system("wget -q https://s3.amazonaws.com/ragab-datasets/imagenet2012/imagenet1k-val.rec -P ~/data/")
    elif dataset == 'imagenet-480px':
        if not os.path.exists(os.path.expanduser('~/data/imagenet_train.rec')):
            os.system("wget -q https://s3.amazonaws.com/aws-ml-platform-datasets/480px-95quality-imagenet/imagenet_train.rec -P ~/data/")
        if not os.path.exists(os.path.expanduser('~/data/imagenet_val.rec')):
            os.system("wget -q https://s3.amazonaws.com/aws-ml-platform-datasets/480px-95quality-imagenet/imagenet_val.rec -P ~/data/")
        if not os.path.exists(os.path.expanduser('~/data/imagenet1k-val.rec')):
            os.system("wget -q https://s3.amazonaws.com/ragab-datasets/imagenet2012/imagenet1k-val.rec -P ~/data/")
    else:
        raise ValueError('Unknown dataset')
# wget -q https://s3.amazonaws.com/aws-ml-platform-datasets/480px-95quality-imagenet/imagenet_train.rec -P data/
# wget -q https://s3.amazonaws.com/aws-ml-platform-datasets/480px-95quality-imagenet/imagenet_val.rec -P data/
# cd ~

#python benchmark_runner.py --task-name tornadomeet-val256 --num-gpus 8 --metrics-policy metrics_parameters_images_top_1_plotacc --metrics-suffix custom.p3_16x --framework mxnet --data-set imagenet-480px --command-to-execute "python image_classification/train_imagenet.py --data-train ~/data/imagenet_train.rec --data-val ~/data/imagenet1k-val.rec --gpus 1,0,2,3,4,5,6,7 --batch-size 512 --data-nthreads 35 --num-epochs 120 --dtype float32 --min-random-scale 0.533 --lr 0.1 --lr-step-epochs 30,60,90"