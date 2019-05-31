# !/bin/python

from collections import defaultdict

import nltk
import numpy as np
from tqdm import tqdm

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

import classify

# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('punkt')
ps = WordNetLemmatizer()
# ps = PorterStemmer()
stop_words = nltk.corpus.stopwords.words('english')


def my_tokenizer(text):
    # text = text.translate(dict.fromkeys(map(ord, string.punctuation), ' '))
    result = [ps.lemmatize(w) for w in word_tokenize(text.lower())]
    return " ".join(result)
    # return [ps.stem(w) for w in word_tokenize(text)]


def read_files(tarfname):
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
    from sklearn.feature_extraction.text import TfidfVectorizer
    # sentiment.count_vect = TfidfVectorizer(tokenizer=my_tokenizer)
    # sentiment.count_vect = TfidfVectorizer(tokenizer=my_tokenizer, stop_words='english')
    sentiment.count_vect = TfidfVectorizer(tokenizer=my_tokenizer, min_df=1, ngram_range=(1, 2))
    sentiment.trainX = sentiment.count_vect.fit_transform(sentiment.train_data)
    sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)

    from sklearn import preprocessing
    sentiment.le = preprocessing.LabelEncoder()
    sentiment.le.fit(sentiment.train_labels)
    sentiment.target_labels = sentiment.le.classes_
    sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
    sentiment.devy = sentiment.le.transform(sentiment.dev_labels)
    tar.close()
    return sentiment


def read_unlabeled(tarfname):

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

# class TfidfEmbeddingVectorizer(object):
#     def __init__(self, w2v):
#         self.w2v = w2v
#         self.word2weight = None
#         self.dim = len(list(w2v.items())[0][1])
#
#     def fit(self, X):
#         tfidf = TfidfVectorizer(analyzer=lambda x: x, ngram_range=(1,2), max_df=1)
#         tfidf.fit(X)
#         max_idf = max(tfidf.idf_)
#         self.word2weight = defaultdict(
#             lambda: max_idf,
#             [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
#
#         return self
#
#     def transform(self, X):
#         return np.array([
#                 np.mean([self.w2v[w] * self.word2weight[w]
#                          for w in words if w in self.w2v] or
#                         [np.zeros(self.dim)], axis=0)
#                 for words in X
#             ])

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

def classifier():
    print("Reading data")
    tarfname = "../data/sentiment/sentiment.tar.gz"
    sentiment = read_files(tarfname)
    # print("\nReading unlabeled data")
    # unlabeled = read_unlabeled(tarfname)

    # print("-- Loading embedding")
    # #GLOVE_6B_50D_PATH = '../data/glove.840B.300d.txt'
    # FAST_TEXT = '../data/crawl-300d-2M.vec'
    # encoding = "utf-8"
    #
    # all_words = set(w for words in sentiment.train_data for w in words)
    # all_words |= set(w for words in sentiment.dev_data for w in words)
    # all_words |= set(w for words in unlabeled.data for w in words)
    #
    # fast_text = {}
    # with open(FAST_TEXT, "r", encoding='utf-8', newline='\n', errors='ignore') as infile:
    #     infile.readline()
    #     for line in tqdm(infile):
    #         parts = line.rstrip().split()
    #         word = parts[0]
    #         if word in all_words:
    #             nums = np.array(parts[1:], dtype=np.float32)
    #             fast_text[word] = nums

    # glove = {}
    # with open(GLOVE_6B_50D_PATH, "rb") as infile:
    #     for line in tqdm(infile):
    #         parts = line.split()
    #         word = parts[0].decode(encoding)
    #         if word in all_words:
    #             nums = np.array(parts[1:], dtype=np.float32)
    #             glove[word] = nums




    # import gensim
    #
    # # let X be a list of tokenized texts (i.e. list of lists of tokens)
    # model = gensim.models.Word2Vec(sentiment.train_data + sentiment.dev_data + unlabeled.data, size=300, min_count=2, iter=15)
    # w2v = dict(zip(model.wv.index2word, model.wv.syn0))

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
    #         #print("\nTraining classifier")
    #
    #         cls = classify.train_classifier(sentiment.trainX, sentiment.trainy, C=0.53)
    #         classify.evaluate(sentiment.trainX, sentiment.trainy, cls, 'train')
    #         classify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev')

    # sentiment.count_vect = EmbeddingVectorizer(fast_text)
    #
    # sentiment.trainX = sentiment.count_vect.transform(sentiment.train_data)
    # sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)
    # unlabeled.X = sentiment.count_vect.transform(unlabeled.data)

    cls = classify.train_classifier(sentiment.trainX, sentiment.trainy, C=0.53)
    classify.evaluate(sentiment.trainX, sentiment.trainy, cls, 'train')
    classify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev')
    return (cls, sentiment)


if __name__ == "__main__":
    classifier()

