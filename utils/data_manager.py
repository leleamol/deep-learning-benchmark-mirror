
import os


def getImagenetData(dataset):
	if dataset == 'imagenet' or dataset == 'imagenet-256px':
	    if not os.path.exists('./data/imagenet1k-train.rec'):
	        os.system("wget -q https://s3.amazonaws.com/ragab-datasets/imagenet2012/imagenet1k-train.rec -P data/")
	    if not os.path.exists('./data/imagenet1k-val.rec'):
	        os.system("wget -q https://s3.amazonaws.com/ragab-datasets/imagenet2012/imagenet1k-val.rec -P data/")
	elif dataset == 'imagenet-480px':
		if not os.path.exists('./data/imagenet_train.rec'):
			os.system("wget -q https://s3.amazonaws.com/aws-ml-platform-datasets/480px-95quality-imagenet/imagenet_train.rec -P data/")
		if not os.path.exists('./data/imagenet_val.rec'):
			os.system("wget -q https://s3.amazonaws.com/aws-ml-platform-datasets/480px-95quality-imagenet/imagenet_val.rec -P data/")
	else:
		raise ValueError('Unknown dataset')
