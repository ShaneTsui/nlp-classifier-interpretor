import tarfile
import re

def read_tsv(tar, fname):
    member = tar.getmember(fname)
    print(member.name)
    tf = tar.extractfile(member)
    data = []
    labels = []
    for line in tf:
        line = line.decode("utf-8")
        (label,text) = line.strip().split("\t")
        labels.append(label)
        data.append(text)
    return data, labels

def strip_formatting(string):
    string = string.lower()
    string = re.sub(r"([.!?,'/()])", r" \1 ", string)
    return string

tar = tarfile.open("../data/sentiment/sentiment.tar.gz", "r:gz")
trainname = "train.tsv"
devname = "dev.tsv"
for member in tar.getmembers():
    if 'train.tsv' in member.name:
        trainname = member.name
    elif 'dev.tsv' in member.name:
        devname = member.name

# print("-- Train_data:")
train_data, train_labels = read_tsv(tar, trainname)
# print(type(train_data))
# print(type(train_data[10]), type(train_labels[10]))
# print(train_data[10], train_labels[10])
# print("-- dev data")
dev_data, dev_labels = read_tsv(tar, devname)



with open("../data/sentiment/sentiment_dataset_training.txt", "w", encoding='utf-8') as output_train:
    train_set = zip(train_data, train_labels)
    for data, label in train_set:
        data = data.replace("\n", " ")
        data = strip_formatting(data)
        fasttext_line = "__label__{} {}".format(label, data)
        output_train.write(fasttext_line + "\n")

with open("../data/sentiment/sentiment_dataset_dev.txt", "w", encoding='utf-8') as output_dev:
    dev_set = zip(dev_data, dev_labels)
    for data, label in dev_set:
        data = data.replace("\n", " ")
        data = strip_formatting(data)
        fasttext_line = "__label__{} {}".format(label, data)
        output_dev.write(fasttext_line + "\n")

