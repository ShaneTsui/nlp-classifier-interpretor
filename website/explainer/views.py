from pathlib import Path

from django.shortcuts import render
from django.template import loader


# Create your views here.
from django.http import HttpResponse
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

def parse_plot_data(explainer):
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
                'backgroundColor': "#FF0000FF",
                'borderColor': "#FF0000FF",
                'borderWidth': 1,
                'data': probs[label]
            } for label in labels]


def index(request):
    context = dict()

    if request.method == 'POST':
        text = request.POST.get('text')
        text = text.replace("\n", " ").replace("\r", " ")
        print(text)
        exp = explain_sentence(text)
        out = save_to_file(exp, show_predicted_value=False)
        labels, words, probs = parse_plot_data(exp)
        context["my_data"] = {'labels': words[labels[0]], 'datasets': build_datasets(labels, probs)}
        context["out"] = out
    return render(request, 'explainer/index.html', context)