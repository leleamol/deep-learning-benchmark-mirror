
import os


def getImagenetData():
    

    if not os.path.exists('./data/imagenet1k-train.rec'):
        os.system("wget -q https://s3.amazonaws.com/ragab-datasets/imagenet2012/imagenet1k-train.rec -P data/")

    if not os.path.exists('./data/imagenet1k-val.rec'):
        os.system("wget -q https://s3.amazonaws.com/ragab-datasets/imagenet2012/imagenet1k-val.rec -P data/")
