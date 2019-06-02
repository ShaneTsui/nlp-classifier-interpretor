import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import tarfile

from util.text_util import *

tqdm.pandas()

def read_files(tarfname):
    tar = tarfile.open(tarfname, "r:gz")
    trainname = "train.tsv"
    devname = "dev.tsv"
    for member in tar.getmembers():
        if 'train.tsv' in member.name:
            trainname = member.name
        elif 'dev.tsv' in member.name:
            devname = member.name
    train_data, train_labels = read_tsv(tar, trainname)
    dev_data, dev_labels = read_tsv(tar, devname)



def read_tsv(tar, fname):
    member = tar.getmember(fname)
    tf = tar.extractfile(member)
    data = []
    labels = []
    for line in tf:
        line = line.decode("utf-8")
        (label, text) = line.strip().split("\t")
        if label == "NEGATIVE":
            label = 1
        else: label = 2
        labels.append(label)
        data.append(text)
    return data, labels



def load_data(path):
    print('loading Sentiment data...')
    tar = tarfile.open(path, "r:gz")
    trainname = "train.tsv"
    devname = "dev.tsv"
    for member in tar.getmembers():
        if 'train.tsv' in member.name:
            trainname = member.name
        elif 'dev.tsv' in member.name:
            devname = member.name
    train_data, train_labels = read_tsv(tar, trainname)
    dev_data, dev_labels = read_tsv(tar, devname)

    df_train = pd.DataFrame({"text": train_data, "stars": train_labels}, columns=['text', 'stars'])
    df_dev = pd.DataFrame({"text": dev_data, "stars": dev_labels}, columns=['text', 'stars'])

    df_train['text_tokens'] = df_train['text'].progress_apply(lambda x: normalize(x))
    df_dev['text_tokens'] = df_dev['text'].progress_apply(lambda x: normalize(x))


    train_x, train_y = chunk_to_arrays(df_train)
    train_y = to_one_hot(train_y, dim=2)

    dev_x, dev_y = chunk_to_arrays(df_dev)
    dev_y = to_one_hot(dev_y, dim=2)
    print('finished loading sentiment data')

    return (train_x, train_y), (dev_x, dev_y)

# (train_x, train_y), (test_x, test_y) = load_data("../data/sentiment/sentiment.tar.gz")
# print(type(train_x), type(train_y))
# print(len(train_x), len(test_x))
# print(train_x[:10])
# print(train_y[:10])


