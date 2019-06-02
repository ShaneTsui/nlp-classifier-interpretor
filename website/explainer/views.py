from pathlib import Path

from django.shortcuts import render
from django.template import loader


# Create your views here.
from django.http import HttpResponse

from modules.utils import jsonize
from modules.yelp_predict_with_lime import explain_sentence
from modules.html_writer import *

# def index(request):
#     # template = loader.get_template('explainer/index.html')
#     context = {
#         'latest_question_list': [1, 2, 3],
#     }
#
#     if request.method == 'POST':
#         text = request.POST.get('text')
#         text = text.replace("\n", " ").replace("\r", " ")
#         print(text)
#         exp = explain_sentence(text)
#         output_filename = Path(__file__).parent / "index.html"
#         out = save_to_file(exp, output_filename, show_predicted_value=False)
#         context = {"out" : out }
#
#
#     # return HttpResponse(template.render(context, request))
#     return render(request, 'explainer/index.html', context)

def parse_word_confidence_data(explainer):
    # TODO: Currently only support one label
    labels = explainer.available_labels()
    words, probs = dict(), dict()
    for label in labels:
        exp = explainer.as_list(label)
        words[label], probs[label] = [], []
        for word, prob in exp:
            words[label].append(word)
            probs[label].append(prob)
    return labels, words, probs


def build_datasets(labels, probs):
    # TODO: Change color
    return [{
                'label': label,
                'backgroundColor': "#F0FF00C0",
                'borderColor': "#F0FF00C0",
                'borderWidth': 1,
                'data': probs[label]
            } for label in labels]

def build_dataset(probs):
    # TODO: Change color
    return [{
                'label': "All the classes",
                'backgroundColor': "#CE224DC0",
                'borderColor': "#CE224DC0",
                'borderWidth': 1,
                'data': probs
            }]

def parse_sentence_conf(explainer):
    classes = [str(x) for x in explainer.class_names]
    probas = list(explainer.predict_proba.astype(float))
    return classes, probas


def index(request):
    context = dict()

    if request.method == 'POST':
        text = request.POST.get('text')
        text = text.replace("\n", " ").replace("\r", " ")
        print(text)
        exp = explain_sentence(text)
        out = save_to_file(exp, show_predicted_value=False)
        labels, words, probs = parse_word_confidence_data(exp)
        context["wordConfData"] = {'labels': words[labels[0]], 'datasets': build_datasets(labels, probs)}
        classes, cls_proba = parse_sentence_conf(exp)
        context["classConfData"] = {'labels': classes, 'datasets': build_dataset(cls_proba)}
        context["out"] = out
    return render(request, 'explainer/index.html', context)