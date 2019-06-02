import os
from pathlib import Path

import fastText
import re
import lime.lime_text
import numpy as np
import webbrowser


# This function regularizes a piece of text so it's in the same format
# that we used when training the FastText classifier.
def strip_formatting(string):
    string = string.lower()
    string = re.sub(r"([.!?,'/()])", r" \1 ", string)
    return string

# LIME needs to be able to mimic how the classifier splits
# the string into words. So we'll provide a function that
# mimics how FastText works.
def tokenize_string(string):
    return string.split()

# Load our trained FastText classifier model (created in Part 2)
classifier = fastText.load_model('F:/Data/yelp/reviews_model_ngrams.bin')

# Create a LimeTextExplainer. This object knows how to explain a text-based
# prediction by dropping words randomly.
explainer = lime.lime_text.LimeTextExplainer(
    # We need to tell LIME how to split the string into words. We can do this
    # by giving it a function to call to split a string up the same way FastText does it.
    split_expression=tokenize_string,
    # Our FastText classifer uses bigrams (two-word pairs) to classify text. Setting
    # bow=False tells LIME to not assume that our classifier is based on single words only.
    bow=False,
    # To make the output pretty, tell LIME what to call each possible prediction from our model.
    class_names=["No Stars", "1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"],
    verbose=True
)

# LIME is designed to work with clasText with highlighted wordssifiers that generate predictions
# in the same format as Scikit-Learn. It expects every prediction to have
# a probability value for every possible label.
# The default FastText python wrapper generates predictions in a different
# format where it only returns the top N highest likelihood results. This
# code just calls the FastText predict function and then massages it into
# the format that LIME expects (so that LIME will work).
def fasttext_prediction_in_sklearn_format(classifier, texts):
    res = []
    # Ask FastText for the top 10 most likely labels for each piece of text.
    # This ensures we always get a probability score for every possible label in our model.
    # TODO: N = 10, only return top-N highest predictions
    labels, probabilities = classifier.predict(texts, 10)

    # For each prediction, sort the probabaility scores into the same order
    # (I.e. no_stars, 1_star, 2_star, etc). This is needed because FastText
    # returns predicitons sorted by most likely instead of in a fixed order.
    for label, probs, text in zip(labels, probabilities, texts):
        order = np.argsort(np.array(label))
        res.append(probs[order])

    return np.array(res)

# TODO: Add parameters here, including
# num_features, num_samples, top_label
def explain_sentence(review, top_labels=1, num_features=20, num_samples=5000):

    # Review to explain
    #review = "I didn't love this place :( The food wasn't very good and I didn't like the service either. Also, I found a bug in my food."

    # Pre-process the text of the review so it matches the training format
    preprocessed_review = strip_formatting(review)

    # Make a prediction and explain it!
    exp = explainer.explain_instance(
        # The review to explain
        preprocessed_review,
        # The wrapper function that returns FastText predictions in scikit-learn format
        classifier_fn=lambda x: fasttext_prediction_in_sklearn_format(classifier, x),
        # How many labels to explain. We just want to explain the single most likely label.
        top_labels=top_labels,
        # How many words in our sentence to include in the explanation. You can try different values.
        num_features=num_features,
        # How many samples shall the lime algorithm generate
        num_samples=5000
    )

    return exp

    # Save the explanation to an HTML file so it's easy to view.
    # You can also get it to other formats: as_list(), as_map(), etc.
    # See https://lime-ml.readthedocs.io/en/latest/lime.html#lime.explanation.Explanation
    # output_filename = Path(__file__).parent / "explanation.html"
    # exp.save_to_file(output_filename, show_predicted_value=False)
    #
    # # Open the explanation html in our web browser.
    # webbrowser.open(output_filename.as_uri())