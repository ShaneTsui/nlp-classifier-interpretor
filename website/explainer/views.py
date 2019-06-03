from functools import reduce

from django.shortcuts import render
import tensorflow as tf

# Create your views here.
import modules.hnatt.utils as hnatt
from modules.utils import *
from modules.lime.yelp_predict_with_lime import explain_sentence


from modules.lime.html_writer import *

graph = tf.get_default_graph()
h_yelp = hnatt.get_model('yelp')
h_sentiment = hnatt.get_model('sentiment')
yelp_classnames = ["No Stars", "1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"]
sentiment_classnames = ["NEGATIVE", "POSITIVE"]


def parse_word_confidence_data(explainer, selected_task):
    # labels: dataset name
    labels = explainer.available_labels()
    label_names = [sentiment_classnames[i] for i in labels] if selected_task == "Sentiment" else [yelp_classnames[i] for i in labels]
    words, probs = dict(), dict()
    for i, label in enumerate(labels):
        exp = explainer.as_list(label)
        words[label_names[i]], probs[label_names[i]] = [], []
        for word, prob in exp:
            words[label_names[i]].append(word)
            probs[label_names[i]].append(prob)
    return label_names, words, probs


'''
labels: dataset name
probs: {dataset_name: [data]}
'''
def build_datasets(labels, probs, color=None):
    # TODO: Change color
    beautiful_color = get_hex_color()#color if color else get_hex_color()
    datasets = []
    for label in labels:
        color = next(beautiful_color) + "F0" #if not color else get_pos_neg_color(label.upper(), hex=True)
        datasets.append({
            'label': label,
            'backgroundColor': color,
            'borderColor': color,
            'borderWidth': 1,
            'data': probs[label]
        })
    return datasets


def parse_sentence_conf(explainer):
    classes = [str(x) for x in explainer.class_names]
    words = ['Value']
    probs = list(explainer.predict_proba.astype(float))
    probs_dict = dict()
    for i, prob in enumerate(probs):
        probs_dict[classes[i]] = [prob]
    if len(probs) != len(classes):
        classes = classes[:i] + ['Others']
        probs_dict['Others'] = sum(probs[i+1:])
    return classes, words, probs_dict


def parse_word_attention(word_attention, cls_prediction):
    words, word_probs = [], []
    # TODO: Notice we only support one sentence
    word_attention = list(reduce(lambda x, y: x + y, word_attention))
    for word, prob in word_attention:
        words.append(word)
        word_probs.append(prob)
    return words, {cls_prediction: word_probs}

def parse_hnatt_class_prob(class_probs):
    class_names = ['positive', 'negative']
    probs = dict()
    probs['negative'], probs['positive'] = [class_probs[0]], [class_probs[1]]
    return class_names, probs


def prob2color(sentence_probabilities, cls):
    # beautiful_color = get_rgb_color()
    r, g, b = get_pos_neg_color(cls)
    return f"rgba({r}, {g}, {b}, {sentence_probabilities})"


def parse_sentence_prob(splited_sentences, sentence_attention, cls):
    assert len(splited_sentences) == len(sentence_attention)
    sentence_probabilities = dict()
    for sent, prob in zip(splited_sentences, sentence_attention):
        sentence_probabilities[sent] = prob2color(prob, cls)
    return sentence_probabilities


def lime_explain(request):
    context = dict()
    # Explain text with configuration above using lime
    context['test_sentence'] = "Text..."
    context['top_labels_val'] = 1
    context['num_features_val'] = 20
    context['num_samples_val'] = 2000

    if request.method == 'POST':
        # Get data from post request
        text = request.POST.get('text')
        top_labels = int(request.POST.get('top_labels'))
        num_features = int(request.POST.get('num_features'))
        num_samples = int(request.POST.get('num_samples'))
        selected_task = request.POST.get('selectedTask')
        context['task_clicked'] = selected_task
        # Explain text with configuration above using lime
        context['test_sentence'] = text
        context['top_labels_val'] = top_labels
        context['num_features_val'] = num_features
        context['num_samples_val'] = num_samples

        text = text.replace("\n", " ").replace("\r", " ")
        exp = explain_sentence(text, selected_task, top_labels, num_features, num_samples)
        out = save_to_file(exp, show_predicted_value=False)
        labels, words, probs = parse_word_confidence_data(exp, selected_task)
        context["wordConfData"] = {'labels': words[labels[0]], 'datasets': build_datasets(labels, probs)}
        classes, cls_words, cls_proba = parse_sentence_conf(exp)
        context["classConfData"] = {'labels': cls_words, 'datasets': build_datasets(classes, cls_proba)}
        context["out"] = out

    return render(request, 'explainer/lime_exp.html', context)

def get_prediction(cls_prob):
    return "POSITIVE" if cls_prob[1] > cls_prob[0] else "NEGATIVE"


def hnatt_explain(request):
    context = dict()

    if request.method == 'POST':
        # Get data from post request
        text = request.POST.get('text')
        dataset = request.POST.get('selectedTask') #'yelp' # TODO: request.POST.get('dataset')

        # Explain text with configuration above using lime
        context['test_sentence'] = text

        text = text.replace("\n", " ").replace("\r", " ")
        global graph
        with graph.as_default():
            if dataset == 'Yelp':
                exp = hnatt.explain(h_yelp, text)
            elif dataset=='Sentiment':
                exp = hnatt.explain(h_sentiment, text)
            else:
                raise NotImplementedError
        context['task_clicked'] = dataset
        word_attention = exp['word_attention']
        # TODO: colored sentences
        sentence_attention = exp['sentence_attention']
        splited_sentences = exp['splited_sentences']
        class_probs = exp['probs']
        class_pred = get_prediction(class_probs)
        sentence_prob = parse_sentence_prob(splited_sentences, sentence_attention, class_pred)

        context['sentenceProbabilities'] = sentence_prob

        words, word_probs = parse_word_attention(word_attention, class_pred)
        context['wordConfData'] = {'labels': words, 'datasets': build_datasets([class_pred], word_probs, get_pos_neg_color(class_pred, hex=True))}

        class_names, cls_proba = parse_hnatt_class_prob(class_probs)
        context["classConfData"] = {'labels': [''], 'datasets': build_datasets(class_names, cls_proba, get_pos_neg_color(class_pred, hex=True))}
        context["out"] = None
    return render(request, 'explainer/hnatt_exp.html', context)