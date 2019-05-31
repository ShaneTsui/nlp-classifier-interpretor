#!/bin/python
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

def train_classifier(X, y, C, max_iter=10000):
	"""Train a classifier using the given training data.

	Trains logistic regression on the input data with default parameters.
	"""

	# cls = LogisticRegression(random_state=0, solver='lbfgs', max_iter=50, C=6)
	cls = LogisticRegression(random_state=0, solver='sag', max_iter=max_iter, C=C)
	cls.fit(X, y)
	return cls


def evaluate(X, yt, cls, name='data'):
	"""Evaluated a classifier on the given labeled data using accuracy."""
	yp = cls.predict(X)
	acc = metrics.accuracy_score(yt, yp)

	# recall = metrics.recall_score(yt, yp)
	# precision = metrics.precision_score(yt, yp)
	print("  Accuracy on %s  is: %s" % (name, acc))
	# print("  recall on %s  is: %s" % (name, recall))
	# print("  precision on %s  is: %s" % (name, precision))
	return acc


def evaluate_error(X, yt, cls, sentences, name='data'):
	"""Evaluated a classifier on the given labeled data using accuracy."""
	yp = cls.predict(X)
	acc = metrics.accuracy_score(yt, yp)
	recall = metrics.recall_score(yt, yp)
	precision = metrics.precision_score(yt, yp)

	print("  Accuracy on %s  is: %s" % (name, acc))
	print("  recall on %s  is: %s" % (name, recall))
	print("  precision on %s  is: %s" % (name, precision))

	FP = sentences[np.logical_and(yt < 0.5, yp > 0.5)]
	FN = sentences[np.logical_and(yt > 0.5, yp < 0.5)]

	print(FP)
	print("=" * 50)
	print(FN)
	return acc