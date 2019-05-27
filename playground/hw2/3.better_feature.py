# !/bin/python
# import nltk
import io
from collections import defaultdict

import nltk
import numpy as np
from tqdm import tqdm
import string
import gensim
import re
import pickle

import seaborn as sns

import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import classify

nltk.download('wordnet')
ps = WordNetLemmatizer()
# ps = PorterStemmer()
stop_words = nltk.corpus.stopwords.words('english')


def my_tokenizer(text):
    # text = text.translate(dict.fromkeys(map(ord, string.punctuation), ' '))
    return [ps.lemmatize(w) for w in word_tokenize(text.lower())]
    # return [ps.stem(w) for w in word_tokenize(text)]


def read_files(tarfname):
    """Read the training and development data from the sentiment tar file.
    The returned object contains various fields that store sentiment data, such as:

    train_data,dev_data: array of documents (array of words)
    train_fnames,dev_fnames: list of filenames of the doccuments (same length as data)
    train_labels,dev_labels: the true string label for each document (same length as data)

    The data is also preprocessed for use with scikit-learn, as:

    count_vec: CountVectorizer used to process the data (for reapplication on new data)
    trainX,devX: array of vectors representing Bags of Words, i.e. documents processed through the vectorizer
    le: LabelEncoder, i.e. a mapper from string labels to ints (stored for reapplication)
    target_labels: List of labels (same order as used in le)
    trainy,devy: array of int labels, one for each document
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    trainname = "train.tsv"
    devname = "dev.tsv"
    for member in tar.getmembers():
        if 'train.tsv' in member.name:
            trainname = member.name
        elif 'dev.tsv' in member.name:
            devname = member.name

    class Data:
        pass

    sentiment = Data()

    print("-- train data")
    sentiment.raw_train_data, sentiment.train_labels = read_tsv(tar, trainname)
    sentiment.train_data = list(map(my_tokenizer, sentiment.raw_train_data))
    print(len(sentiment.train_data))

    print("-- dev data")
    sentiment.raw_dev_data, sentiment.dev_labels = read_tsv(tar, devname)
    sentiment.dev_data = list(map(my_tokenizer, sentiment.raw_dev_data))
    print(len(sentiment.dev_data))

    print("-- transforming data and labels")
    # from sklearn.feature_extraction.text import CountVectorizer
    # from sklearn.feature_extraction.text import TfidfVectorizer
    # sentiment.count_vect = TfidfVectorizer(tokenizer=my_tokenizer)
    # sentiment.count_vect = TfidfVectorizer(tokenizer=my_tokenizer, stop_words='english')
    # sentiment.count_vect = TfidfVectorizer(tokenizer=my_tokenizer, min_df=1, ngram_range=(2, 3))
    # sentiment.trainX = sentiment.count_vect.fit_transform(sentiment.train_data)
    # sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)

    from sklearn import preprocessing
    sentiment.le = preprocessing.LabelEncoder()
    sentiment.le.fit(sentiment.train_labels)
    sentiment.target_labels = sentiment.le.classes_
    sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
    sentiment.devy = sentiment.le.transform(sentiment.dev_labels)
    tar.close()
    return sentiment


def read_unlabeled(tarfname):
    """Reads the unlabeled data.

    The returned object contains three fields that represent the unlabeled data.

    data: documents, represented as sequence of words
    fnames: list of filenames, one for each document
    X: bag of word vector for each document, using the sentiment.vectorizer
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")

    class Data:
        pass

    unlabeled = Data()
    unlabeled.data = []

    unlabeledname = "unlabeled.tsv"
    for member in tar.getmembers():
        if 'unlabeled.tsv' in member.name:
            unlabeledname = member.name

    print(unlabeledname)
    tf = tar.extractfile(unlabeledname)
    for line in tf:
        line = line.decode("utf-8")
        text = line.strip()
        unlabeled.data.append(my_tokenizer(text))

    tar.close()
    return unlabeled


def read_tsv(tar, fname):
    member = tar.getmember(fname)
    print(member.name)
    tf = tar.extractfile(member)
    data = []
    labels = []
    for line in tf:
        line = line.decode("utf-8")
        (label, text) = line.strip().split("\t")
        labels.append(label)
        data.append(text)
    return data, labels


def write_pred_kaggle_file(unlabeled, cls, outfname, sentiment):
    """Writes the predictions in Kaggle format.

    Given the unlabeled object, classifier, outputfilename, and the sentiment object,
    this function write sthe predictions of the classifier on the unlabeled data and
    writes it to the outputfilename. The sentiment object is required to ensure
    consistent label names.
    """
    yp = cls.predict(unlabeled.X)
    labels = sentiment.le.inverse_transform(yp)
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    for i in range(len(unlabeled.data)):
        f.write(str(i + 1))
        f.write(",")
        f.write(labels[i])
        f.write("\n")
    f.close()


def write_gold_kaggle_file(tsvfile, outfname):
    """Writes the output Kaggle file of the truth.

    You will not be able to run this code, since the tsvfile is not
    accessible to you (it is the test labels).
    """
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    i = 0
    with open(tsvfile, 'r') as tf:
        for line in tf:
            (label, review) = line.strip().split("\t")
            i += 1
            f.write(str(i))
            f.write(",")
            f.write(label)
            f.write("\n")
    f.close()


def write_basic_kaggle_file(tsvfile, outfname):
    """Writes the output Kaggle file of the naive baseline.

    This baseline predicts POSITIVE for all the instances.
    """
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    i = 0
    with open(tsvfile, 'r') as tf:
        for line in tf:
            (label, review) = line.strip().split("\t")
            i += 1
            f.write(str(i))
            f.write(",")
            f.write("POSITIVE")
            f.write("\n")
    f.close()

class TfidfEmbeddingVectorizer(object):
    def __init__(self, w2v):
        self.w2v = w2v
        self.word2weight = None
        self.dim = len(list(w2v.items())[0][1])

    def fit(self, X):
        tfidf = TfidfVectorizer(analyzer=lambda x: x, ngram_range=(1,2), max_df=1)
        tfidf.fit(X)
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.w2v[w] * self.word2weight[w]
                         for w in words if w in self.w2v] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])

class EmbeddingVectorizer(object):
    def __init__(self, w2v):
        self.w2v = w2v
        self.dim = len(list(w2v.items())[0][1])
        print(self.dim)

    def transform(self, X):
        res = []
        for words in X:
            res.append(np.mean([self.w2v[w] for w in words if w in self.w2v] or [np.zeros(self.dim)], axis=0))
        res = np.array(res)
        print(res.shape)
        return res

if __name__ == "__main__":
    print("Reading data")
    tarfname = "data/sentiment.tar.gz"
    sentiment = read_files(tarfname)
    print("\nReading unlabeled data")
    unlabeled = read_unlabeled(tarfname)

    print("-- Loading embedding")
    GLOVE_6B_50D_PATH = 'D:\Dataset\glove\glove.6B\glove.6B.50d.txt'
    GLOVE_6B_50D_PATH = 'D:\Dataset\glove\glove.840B.300d.txt'
    FAST_TEXT = 'D:\Dataset/fasttext/crawl-300d-2M.vec'
    encoding = "utf-8"

    all_words = set(w for words in sentiment.train_data for w in words)
    all_words |= set(w for words in sentiment.dev_data for w in words)
    all_words |= set(w for words in unlabeled.data for w in words)

    fast_text = {}
    with open(FAST_TEXT, "r", encoding='utf-8', newline='\n', errors='ignore') as infile:
        infile.readline()
        for line in tqdm(infile):
            parts = line.rstrip().split()
            word = parts[0]
            if word in all_words:
                nums = np.array(parts[1:], dtype=np.float32)
                fast_text[word] = nums

    # glove = {}
    # with open(GLOVE_6B_50D_PATH, "rb") as infile:
    #     for line in tqdm(infile):
    #         parts = line.split()
    #         word = parts[0].decode(encoding)
    #         if word in all_words:
    #             nums = np.array(parts[1:], dtype=np.float32)
    #             glove[word] = nums
    #
    #
    # import gensim
    #
    # # let X be a list of tokenized texts (i.e. list of lists of tokens)
    # model = gensim.models.Word2Vec(sentiment.train_data + sentiment.dev_data + unlabeled.data, size=300, min_count=2, iter=15)
    # w2v = dict(zip(model.wv.index2word, model.wv.syn0))
    #
    # for ebd_name, embedding in [("fast_text", fast_text), ("glove", glove), ("word2vec", w2v)]:
    #     for type in ["mean", "tfidf"]:
    #         print(f"-- Embedding {ebd_name} {type}")
    #         if type == "mean":
    #             sentiment.count_vect = EmbeddingVectorizer(embedding)
    #         else:
    #             sentiment.count_vect = TfidfEmbeddingVectorizer(embedding)
    #             sentiment.count_vect.fit(sentiment.train_data)
    #         sentiment.trainX = sentiment.count_vect.transform(sentiment.train_data)
    #         sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)
    #         unlabeled.X = sentiment.count_vect.transform(unlabeled.data)
    #
    #         print("\nTraining classifier")
    #
    #         X, Yt, Yd = [], [], []
    #         for C in range(1, 121, 2):
    #             x = C / 100
    #             # x = C
    #             print(x)
    #             X.append(x)
    #             cls = classify.train_classifier(sentiment.trainX, sentiment.trainy, C=x)
    #             Yt.append(classify.evaluate(sentiment.trainX, sentiment.trainy, cls, 'train'))
    #             Yd.append(classify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev'))
    #
    #         # plt.plot(X, Yt, linewidth=1, label="train")
    #         plt.plot(X, Yd, linewidth=1, label=f"{ebd_name}+{type}")
    #
    # plt.xlabel("C")
    # plt.ylabel("accuracy")
    # plt.legend()
    # plt.savefig("data/acc_new_feature")
    # plt.show()

    sentiment.count_vect = EmbeddingVectorizer(fast_text)
    sentiment.trainX = sentiment.count_vect.transform(sentiment.train_data)
    sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)
    # unlabeled.X = sentiment.count_vect.transform(unlabeled.data)

    cls = classify.train_classifier(sentiment.trainX, sentiment.trainy, C=0.65)
    classify.evaluate(sentiment.trainX, sentiment.trainy, cls, 'train')
    classify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev')
    classify.evaluate_error(sentiment.devX, sentiment.devy, cls, np.array(sentiment.raw_dev_data), 'dev')

    # print("Writing predictions to a file")
    # write_pred_kaggle_file(unlabeled, cls, "data/sentiment-pred.csv", sentiment)