import fastText
import re
import os
import lime.lime_text
import numpy as np
import webbrowser
from pathlib import Path, PurePosixPath
import sys

model_dir = "../data/sentiment/"

def strip_formatting(string):
    string = string.lower()
    string = re.sub(r"([.!?,'/()])", r" \1 ", string)
    return string


def tokenize_string(string):
    return string.split()


classifier = fastText.load_model(model_dir + 'sentiment_model_ngrams.bin')

explainer = lime.lime_text.LimeTextExplainer(
    split_expression=tokenize_string,
    bow=False,
    class_names=["NEGATIVE", "POSITIVE"]
)

def fasttext_prediction_in_sklearn_format(classifier, texts):
    res = []

    labels, probabilities = classifier.predict(texts, 10)

    for label, probs, text in zip(labels, probabilities, texts):
        order = np.argsort(np.array(label))
        res.append(probs[order])

    return np.array(res)


def main(review):
    preprocessed_review = strip_formatting(review)
    # Make a prediction and explain it!
    exp = explainer.explain_instance(
        # The review to explain
        preprocessed_review,
        # The wrapper function that returns FastText predictions in scikit-learn format
        classifier_fn=lambda x: fasttext_prediction_in_sklearn_format(classifier, x),
        # How many labels to explain. We just want to explain the single most likely label.
        top_labels=1,
        # How many words in our sentence to include in the explanation. You can try different values.
        num_features=20,
    )

    # Save the explanation to an HTML file so it's easy to view.
    # You can also get it to other formats: as_list(), as_map(), etc.
    # See https://lime-ml.readthedocs.io/en/latest/lime.html#lime.explanation.Explanation
    output_filename = os.getcwd() + "/explanation.html"
    exp.save_to_file("./explanation.html")

    # Open the explanation html in our web browser.
    uri = PurePosixPath(output_filename).as_uri()
    webbrowser.open(uri)


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("""
    #     Usage: python explanation_sentiment.py [text]\n
    #            Generate rating prediction and explanation for the review you entered.\n""")
    #     sys.exit(1)
    # main(sys.argv[1])

    main("It is a good place to have fun")
