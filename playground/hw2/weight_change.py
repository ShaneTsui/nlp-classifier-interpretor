#!/bin/python
# import nltk
import nltk
import random
from tqdm import tqdm
import scipy, time
import numpy as np
import scipy.sparse as sp
import string
import re
import goslate
gs = goslate.Goslate()

import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet

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
    sentiment.train_data, sentiment.train_labels = read_tsv(tar, trainname)
    print(len(sentiment.train_data))

    print("-- dev data")
    sentiment.dev_data, sentiment.dev_labels = read_tsv(tar, devname)
    print(len(sentiment.dev_data))

    print("-- transforming data and labels")
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    sentiment.count_vect = CountVectorizer()
    sentiment.trainCX = sentiment.count_vect.fit_transform(sentiment.train_data)
    sentiment.devCX = sentiment.count_vect.transform(sentiment.dev_data)

    sentiment.tfidf_vect = TfidfVectorizer(tokenizer=my_tokenizer, min_df=1, ngram_range=(1, 2))
    sentiment.trainX = sentiment.tfidf_vect.fit_transform(sentiment.train_data)
    sentiment.devX = sentiment.tfidf_vect.transform(sentiment.dev_data)

    from sklearn import preprocessing
    sentiment.le = preprocessing.LabelEncoder()
    sentiment.le.fit(sentiment.train_labels)
    sentiment.target_labels = sentiment.le.classes_
    sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
    sentiment.devy = sentiment.le.transform(sentiment.dev_labels)
    tar.close()
    return sentiment


def read_unlabeled(tarfname, sentiment):
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
        unlabeled.data.append(text)

    unlabeled.X = sentiment.tfidf_vect.transform(unlabeled.data)
    print(unlabeled.X.shape)
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


def delete_rows_csr(mat, indices):
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[list(indices)] = False
    return mat[mask]


if __name__ == "__main__":
    print("Reading data")
    tarfname = "data/sentiment.tar.gz"
    sentiment = read_files(tarfname)
    unlabeled = read_unlabeled(tarfname, sentiment)

    def semi_learning(delabel_percentage):
        thresh_hold = 4
        train_total_acc, dev_total_acc, baseline_acc = 0, 0, 0
        T = 1
        np.random.seed(int(time.time()))

        for i in range(T):
            # rng = np.random.RandomState()
            delabel_idx = np.random.rand(unlabeled.X.shape[0]) <= delabel_percentage
            unlabeledX = scipy.sparse.csr_matrix.copy(unlabeled.X[delabel_idx])
            print(f"unlabeledX.shape = {unlabeledX.shape}")
            epoch = 0
            trainx, trainy = sentiment.trainX, sentiment.trainy
            cls = classify.train_classifier(trainx, trainy, C=5)
            baseline_acc += classify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev')
            best_trainx, best_trainy, best_acc, best_cls = None, None, 0, None
            last_num, last_pred, patience = 0, 0, 7
            t = 0

            while trainx.shape[0] - last_num > 0:
                cls = classify.train_classifier(trainx, trainy, C=5)

                # print("\nEvaluating")
                classify.evaluate(sentiment.trainX, sentiment.trainy, cls, 'train')
                dev_acc = classify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev')
                if dev_acc > best_acc:
                    best_cls = cls
                    best_trainx, best_trainy, best_acc = scipy.sparse.csr_matrix.copy(trainx), np.copy(trainy), dev_acc
                    t = 0
                else:
                    t += 1
                if t > patience:
                    break
                # Select top-k best item from unlabeled data
                score = np.abs(cls.decision_function(unlabeledX))
                good_pred_X = unlabeledX[score > thresh_hold]
                if not good_pred_X.shape[0]:
                    break
                good_pred_y = cls.predict(unlabeledX[score > thresh_hold])
                unlabeledX = delete_rows_csr(unlabeledX, score > thresh_hold)
                last_num = trainx.shape[0]

                trainx = sp.vstack((trainx, good_pred_X), format='csr')
                trainy = np.concatenate((trainy, good_pred_y), axis=0)

                # print(f"{trainx.shape[0]} data after {epoch} epoch")
            cls = best_cls
            new_voc2idx = sentiment.tfidf_vect.vocabulary_
            new_idx2voc = {idx: word for word, idx in sentiment.tfidf_vect.vocabulary_.items()}
            print(len(new_idx2voc))
            new_weights = cls.coef_.reshape(-1)
            # new_weight_idx = np.argsort(new_weights).reshape(-1)
            # neg = []
            # for w in new_weight_idx[:20]:
            #     neg.append(new_idx2voc[w])
            # print(", ".join(neg))
            #
            # pos = []
            # for w in new_weight_idx[-20:][::-1]:
            #     pos.append(new_idx2voc[w])
            # print(", ".join(pos))

            base_cls = classify.train_classifier(sentiment.trainCX, sentiment.trainy, C=1)
            base_voc2idx = sentiment.count_vect.vocabulary_
            base_idx2voc = {idx: word for word, idx in sentiment.count_vect.vocabulary_.items()}
            print(len(base_idx2voc))
            base_weights = base_cls.coef_.reshape(-1)

            words, weight_delta = [], []
            for word, idx in tqdm(base_voc2idx.items()):
                if word not in new_voc2idx:
                    # print(word)
                    continue
                words.append(word)
                weight_delta.append(new_weights[new_voc2idx[word]] - base_weights[base_voc2idx[word]])
            words = np.array(words)
            weight_delta = np.array(weight_delta)
            sorted_idx = np.argsort(weight_delta)
            most_changed_words = []

            sample_num = 15
            neg = []
            for w in sorted_idx[:sample_num]:
                neg.append(words[w])
            print(" & ".join(neg))
            print(" & ".join(str(round(new_weights[new_voc2idx[word]], 1)) for word in neg))
            print(" & ".join(str(round(base_weights[base_voc2idx[word]], 1)) for word in neg))
            print(" & ".join(str(round(weight_delta[idx], 1)) for idx in sorted_idx[:sample_num]))


            pos = []
            for w in sorted_idx[-sample_num:][::-1]:
                pos.append(words[w])
            print(" & ".join(pos))
            print(" & ".join(str(round(new_weights[new_voc2idx[word]], 1)) for word in pos))
            print(" & ".join(str(round(base_weights[base_voc2idx[word]], 1)) for word in pos))
            print(" & ".join(str(round(weight_delta[idx], 1)) for idx in sorted_idx[-sample_num:][::-1]))

            # for w in sorted_idx[-30:]:
            #     most_changed_words.append(words[w])
            # print(", ".join(most_changed_words))

            # cls = classify.train_classifier(sentiment.trainX, sentiment.trainy, C=15.8)
            # classify.evaluate(sentiment.trainX, sentiment.trainy, cls, 'train')
            # classify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev')

            # print("\nReading unlabeled data")
            # unlabeled = read_unlabeled(tarfname, sentiment)
            # print("Writing predictions to a file")
            # write_pred_kaggle_file(unlabeled, cls, "data/semi-sentiment-pred.csv", sentiment)

            # cls = classify.train_classifier(best_trainx, best_trainy, C=5)
            train_total_acc += classify.evaluate(sentiment.trainX, sentiment.trainy, best_cls, 'train')
            dev_total_acc += classify.evaluate(sentiment.devX, sentiment.devy, best_cls, 'dev')
            # classify.evaluate_error(sentiment.devX, sentiment.devy, best_cls, np.array(sentiment.dev_data), 'dev')
            write_pred_kaggle_file(unlabeled, best_cls, f"data/semi-sentiment-pred-{delabel_percentage}.csv", sentiment)
        print(train_total_acc / T, dev_total_acc / T)
        return train_total_acc / T, dev_total_acc / T, baseline_acc / T

    percentage = [i / 10 for i in range(10, 11)]
    train_acc, dev_acc, baseline_acc = [], [], []
    for p in tqdm(percentage):
        ta, da, ba = semi_learning(p)
        train_acc.append(ta)
        dev_acc.append(da)
        baseline_acc.append(ba)
        print("="*50)
    print(train_acc, dev_acc)
    # plt.plot([p*unlabeled.X.shape[0] for p in percentage], baseline_acc, linewidth=1, label="supervised")
    # plt.plot([p*unlabeled.X.shape[0] for p in percentage], dev_acc, linewidth=1, label="semi-supervised")
    # plt.xlabel("amount of unlabeled data")
    # plt.ylabel("accuracy")
    # plt.legend()
    # plt.savefig("data/acc_semi")
    # plt.show()