from pathlib import Path

from django.shortcuts import render
from django.template import loader
import randomcolor

# Create your views here.
from django.http import HttpResponse

from modules.utils import *
from modules.yelp_predict_with_lime import explain_sentence
from modules.html_writer import *
from chartjs.colors import next_color

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

# color_picker = next_color()
# rand_color = randomcolor.RandomColor()




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
    beautiful_color = get_color()
    datasets = []
    for label in labels:
        color = next(beautiful_color) + "F0"
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
    if len(probs) != classes:
        classes = classes[:i] + ['Others']
        probs_dict['Others'] = sum(probs[i+1:])
    return classes, words, probs_dict


def index(request):
    context = dict()

    if request.method == 'POST':
        # Get data from post request
        text = request.POST.get('text')
        top_labels = int(request.POST.get('top_labels'))
        num_features = int(request.POST.get('num_features'))
        num_samples = int(request.POST.get('num_samples'))

        # Explain text with configuration above using lime
        context['test_sentence'] = text
        context['top_labels_val'] = top_labels
        context['num_features_val'] = num_features
        context['num_samples_val'] = num_samples

        text = text.replace("\n", " ").replace("\r", " ")
        # print(text)
        exp = explain_sentence(text, top_labels, num_features, num_samples)
        out = save_to_file(exp, show_predicted_value=False)
        labels, words, probs = parse_word_confidence_data(exp)
        context["wordConfData"] = {'labels': words[labels[0]], 'datasets': build_datasets(labels, probs)}
        classes, cls_words, cls_proba = parse_sentence_conf(exp)
        context["classConfData"] = {'labels': cls_words, 'datasets': build_datasets(classes, cls_proba)}
        context["out"] = out
    return render(request, 'explainer/index.html', context)