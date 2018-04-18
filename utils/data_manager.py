
import os


def getImagenetData():
    

    os.system('mkdir -p data')
    if not os.path.exists('./data/imagenet1k-train.rec'):
        os.system("aws s3 cp  s3://imagenet-rec/imagenet1k-train.rec data/")
    
    if not os.path.exists('./data/imagenet1k-val.rec'):
        os.system("aws s3 cp  s3://imagenet-rec/imagenet1k-val.rec data/")
~
