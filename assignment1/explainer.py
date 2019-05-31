import os
import lime.lime_text
import webbrowser
from pathlib import PurePosixPath
from sentiment import classifier
from sentiment import my_tokenizer


classifier, sentiment = classifier()
explainer = lime.lime_text.LimeTextExplainer(
    bow=False,
    class_names=["NEGATIVE", "POSITIVE"]
)


def prediction_in_skformat(classifier, texts):
    X = sentiment.count_vect.transform(map(my_tokenizer, texts))
    return classifier.predict_proba(X)


def main(review):
    # Make a prediction and explain it!
    exp = explainer.explain_instance(
        # The review to explain
        review,
        # The wrapper function that returns FastText predictions in scikit-learn format
        classifier_fn=lambda x: prediction_in_skformat(classifier, x),
        # How many labels to explain. We just want to explain the single most likely label.
        top_labels=1,
        # How many words in our sentence to include in the explanation. You can try different values.
        num_features=10,
    )

    # Save the explanation to an HTML file so it's easy to view.
    # You can also get it to other formats: as_list(), as_map(), etc.
    # See https://lime-ml.readthedocs.io/en/latest/lime.html#lime.explanation.Explanation
    output_filename = os.getcwd() + "/_explanation1.html"
    exp.save_to_file("./_explanation1.html")

    # Open the explanation html in our web browser.
    uri = PurePosixPath(output_filename).as_uri()
    webbrowser.open(uri)

if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("""
    #     Usage: python explanation.py [text]\n
    #            Generate rating prediction and explanation for the review you entered.\n""")
    #     sys.exit(1)
    # main(sys.argv[1])
    main("I love this place . My wife and I stop and have breakfast after hiking. Dogs are allowed outside on the patio and they bring water bowls for them. Oh...and more")

