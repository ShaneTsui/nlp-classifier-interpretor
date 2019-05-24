from classify import *
from file_utils import *



if __name__ == "__main__":
    print("Reading data")
    tarfname = "data/sentiment.tar.gz"
    sentiment = read_files(tarfname)

    print("\nTraining classifier")
    import classify
    cls = classify.train_classifier(sentiment.trainX, sentiment.trainy, C=15.8)

    from sklearn.pipeline import make_pipeline

    c = make_pipeline(sentiment.count_vect, cls)

    print(c.predict_proba([sentiment.dev_data[0]]))

    from lime.lime_text import LimeTextExplainer

    class_names = ["Neg", "Pos"]
    lbl2idx = {"NEGATIVE":0, "POSITIVE": 1}
    explainer = LimeTextExplainer(class_names=class_names)

    idx = 83
    exp = explainer.explain_instance(sentiment.dev_data[idx], c.predict_proba, num_features=6)
    print('Document id: %d' % idx)
    print('Probability(POSITIVE) =', c.predict_proba([sentiment.dev_data[idx]])[0, 1])
    print('True class: %s' % class_names[lbl2idx[sentiment.dev_labels[idx]]])

    exp.as_list()

    print('Original prediction:', cls.predict_proba(sentiment.dev_data[idx])[0, 1])
    tmp = sentiment.dev_data[idx].copy()
    # tmp[0, vectorizer.vocabulary_['Posting']] = 0
    # tmp[0, vectorizer.vocabulary_['Host']] = 0
    # print('Prediction removing some features:', rf.predict_proba(tmp)[0, 1])
    # print('Difference:', rf.predict_proba(tmp)[0, 1] - rf.predict_proba(test_vectors[idx])[0, 1])

    # %matplotlib inline
    fig = exp.as_pyplot_figure()
    plt.show()

    exp.show_in_notebook(text=False)
    exp.save_to_file('oi.html')
    exp.show_in_notebook(text=True)


    # print("\nEvaluating")
    # classify.evaluate(sentiment.trainX, sentiment.trainy, cls, 'train')
    #
    # classify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev')
    # classify.evaluate_error(sentiment.devX, sentiment.devy, cls, np.array(sentiment.dev_data), 'dev')