import string
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
import numpy as np

STOP_WORDS = ['the', 'a', 'an']
nlp = English()
nlp.add_pipe(nlp.create_pipe('sentencizer'))

def normalize(text):
	text = text.lower().strip()
	doc = nlp(text)
	filtered_sentences = []
	for sentence in doc.sents:
		filtered_tokens = list()
		for i, w in enumerate(sentence):
			s = w.string.strip()
			if len(s) == 0 or s in string.punctuation and i < len(doc) - 1:
				continue
			if s not in STOP_WORDS:
				s = s.replace(',', '.')
				filtered_tokens.append(s)
		filtered_sentences.append(' '.join(filtered_tokens))
	return filtered_sentences


def chunk_to_arrays(chunk, binary=False):
	x = chunk['text_tokens'].values
	if binary:
		y = chunk['polarized_stars'].values
	else:
		y = chunk['stars'].values
	return x, y

def balance_classes(x, y, dim, train_ratio):
	x_negative = x[np.where(y == 1)]
	y_negative = y[np.where(y == 1)]
	x_positive = x[np.where(y == 2)]
	y_positive = y[np.where(y == 2)]

	n = min(len(x_negative), len(x_positive))
	train_n = int(round(train_ratio * n))
	train_x = np.concatenate((x_negative[:train_n], x_positive[:train_n]), axis=0)
	train_y = np.concatenate((y_negative[:train_n], y_positive[:train_n]), axis=0)
	test_x = np.concatenate((x_negative[train_n:], x_positive[train_n:]), axis=0)
	test_y = np.concatenate((y_negative[train_n:], y_positive[train_n:]), axis=0)

	# import pdb; pdb.set_trace()
	return (train_x, to_one_hot(train_y, dim=2)), (test_x, to_one_hot(test_y, dim=2))

def to_one_hot(labels, dim=5):
	results = np.zeros((len(labels), dim))
	for i, label in enumerate(labels):
		results[i][label - 1] = 1
	return results

def polarize(v):
	if v >= 3:
		return 2
	else:
		return 1