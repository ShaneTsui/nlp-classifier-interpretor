import util.yelp as yelp
import util.sentiment as sentiment
from hnatt import HNATT
from util.text_util import normalize

YELP_DATA_PATH = '../data/yelp/review.json'
SENTIMENT_DATA_PATH = '../data/sentiment/sentiment.tar.gz'

SENT_SAVED_MODEL_DIR = 'saved_models/sentiment/'
YELP_SAVED_MODEL_DIR  = 'saved_models/yelp/'

SAVED_MODEL_FILENAME = 'model.h5'

def find_confidence_cases():
	"""

	:param h: pretrained HNATT model
	:param dev_x: np.array. dev data set
	:param dev_y: np.array. dev label set.
	:param threshold: determine confidence
	:return: list[sentence]
	"""
	_, (dev_x, dev_y) = sentiment.load_data(path=SENTIMENT_DATA_PATH)
	h = get_model('sentiment')
	threshold = 0.9

	with open("../data/sentiment/overconfident.txt", 'w') as over_confident_file:
		with open("../data/sentiment/confident.txt", 'w') as confident_file:
			cases = list(zip(dev_x, dev_y))
			for text, label in cases:
				preds = h.predict([text])[0]
				if max(preds) >= threshold:
					if label[0] == 0:
						label = 'NEGATIVE'
					else:
						label = 'POSITIVE'

					line = label + "\t" + " ".join(text) + "\n"
					#print(line)
					if (preds[0] > preds[1] and label == 'POSITIVE') or (preds[0] < preds[1] and label == 'NEGATIVE'):
					    confident_file.writelines(line)
					else:
						over_confident_file.writelines(line)


def hnatt_explain(testcase, dataset='sentiment', label = ""):
	"""

	:param dataset: string: sentiment/yelp
	:param testcase: string: a review you want to test and visualize
	:return: dict('sentence', 'probs', 'ground_truth','word_attention', 'sentence_attention')

	"""
	h = get_model(dataset)
	dic = {}
	dic['sentence'] = testcase
	preds = h.predict([testcase])
	dic['probs'] = preds[0]
	if label:
		dic["ground_truth"] = label
	else:
		dic["ground_truth"] = "NOT GIVEN"

	activation_maps = h.activation_maps(testcase)
	sentence_probs = []
	word_probs = []
	for sent in activation_maps:
		word_probs.append(sent[0])
		sentence_probs.append(sent[1])
	dic['word_attention'] = word_probs
	dic['sentence_attention'] = sentence_probs
	dic['splited_sentences'] = normalize(testcase)
	return dic

def train(dataset):
	#dataset: string. 'yelp' or 'sentiment'
	if dataset == "yelp":
		(train_x, train_y), (test_x, test_y) = yelp.load_data(path=YELP_DATA_PATH, size=1e4, binary=False)
		SAVED_MODEL_DIR = YELP_SAVED_MODEL_DIR
	else:
		(train_x, train_y), (test_x, test_y) = sentiment.load_data(path=SENTIMENT_DATA_PATH)
		SAVED_MODEL_DIR = SENT_SAVED_MODEL_DIR

	h = HNATT()
	h.train(train_x, train_y,
		batch_size=16,
		epochs=16,
		embeddings_path=None,
		saved_model_dir=SAVED_MODEL_DIR,
		saved_model_filename=SAVED_MODEL_FILENAME)

def get_model(dataset):
	h = HNATT()
	if dataset == "yelp":
		h.load_weights(YELP_SAVED_MODEL_DIR, SAVED_MODEL_FILENAME)
	else:
		h.load_weights(SENT_SAVED_MODEL_DIR, SAVED_MODEL_FILENAME)
	return h



if __name__ == '__main__':
	# train("yelp")
	# train("sentiment")

	dataset = 'sentiment'
	testcase = 'i agree that the seating is odd. but the food is exceptional especially for the price. the menu is truly montreal meats japan (spelling is correct) = very unique. great'
	result = hnatt_explain(dataset, testcase)
	for key in result:
	     print(key, result[key])



	#find_confidence_cases()
