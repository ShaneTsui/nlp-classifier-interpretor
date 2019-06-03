import pandas as pd
import numpy as np
from tqdm import tqdm
import json

from modules.hnatt.util.text_util import *

tqdm.pandas()

def load_data(path, size=1e4, train_ratio=0.8, binary=False):
	print('loading Yelp reviews...')
	#df = pd.read_csv(path, nrows=size, usecols=['stars', 'text'])
	text = []
	stars = []
	with open(path) as f:
		inputs = f.readlines()
		i = 0
		for line in tqdm(inputs):
			if i > size:
				break
			review_data = json.loads(line)
			text.append(review_data['text'].replace("\n", " "))
			stars.append(int(review_data['stars']))
			i += 1

	df = pd.DataFrame({"text": text, "stars": stars}, columns=['text', 'stars'])

	df['text_tokens'] = df['text'].progress_apply(lambda x: normalize(x))
	
	dim = 5
	if binary:
		dim = 2

	if binary:
		df['polarized_stars'] = df['stars'].apply(lambda x: polarize(x))
		x, y = chunk_to_arrays(df, binary=binary)
		return balance_classes(x, y, dim, train_ratio)

	train_size = round(size * train_ratio)
	print("train_size:", train_size)
	test_size = size - train_size
	print("test_size", test_size)

	# training + validation set
	train_x = np.empty((0,))
	train_y = np.empty((0,))

	train_set = df[0:train_size].copy()
	train_set['len'] = train_set['text_tokens'].apply(lambda x: len(x))
	# train_set.sort_values('len', inplace=True, ascending=True)
	train_x, train_y = chunk_to_arrays(train_set, binary=binary)
	train_y = to_one_hot(train_y, dim=dim)

	test_set = df[train_size:]
	test_x, test_y = chunk_to_arrays(test_set, binary=binary)
	test_y = to_one_hot(test_y)
	print('finished loading Yelp reviews')

	return (train_x, train_y), (test_x, test_y)


#(train_x, train_y), (test_x, test_y) = load_data("../data/yelp/review2.json", size=40, train_ratio=0.2, binary=False)
