import fastText
import re

model_path = "../data/sentiment/"

def strip_formatting(string):
    string = string.lower()
    string = re.sub(r"([.!?,'/()])", r" \1 ", string)
    return string

# Reviews to check
reviews = [
    "i drove 10 miles at 730pm and found a closed shop. hours say till 9pm, so thanks for wasting my time and gas.",
    "I hate this place so much. They were mean to me.",
    "I don't know. It was ok, I guess. Not really sure what to say.",
    "I love this place! The food is very delicious."
]

# Pre-process the text of each review so it matches the training format
preprocessed_reviews = list(map(strip_formatting, reviews))

# Load the model
classifier = fastText.load_model(model_path + 'sentiment_model.bin')

# Get fastText to classify each review with the model
labels, probabilities = classifier.predict(preprocessed_reviews, 1)

# Print the results
for review, label, probability in zip(reviews, labels, probabilities):

    print("{} ({}% confidence)".format(label, int(probability[0] * 100)))
    print(review)
    print()