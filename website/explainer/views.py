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
import website.settings as settings


def parse_word_confidence_data(explainer):
    # labels: dataset name
    labels = explainer.available_labels()
    words, probs = dict(), dict()
    for label in labels:
        exp = explainer.as_list(label)
        words[label], probs[label] = [], []
        for word, prob in exp:
            words[label].append(word)
            probs[label].append(prob)
    return labels, words, probs

'''
labels: dataset name
probs: {dataset_name: [data]}
'''
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

    return render(request, 'explainer/lime_exp.html', context)


def parse_word_attention(word_attention):
    words, word_probs = [], []
    for word, prob in word_attention:
        words.append(word)
        word_probs.append(prob)
    return ['word attention'], words, {'word attention': word_probs}

def parse_hnatt_class_prob(class_probs):
    class_names = ['negative', 'positive']
    probs = dict()
    probs['negative'], probs['positive'] = [class_probs[0]], [class_probs[1]]
    return class_names, probs

def hnatt_explain(request):
    context = dict()

    if request.method == 'POST':
        # Get data from post request
        text = "i agree that the seating is odd. but the food is exceptional especially for the price. the menu is truly montreal meats japan (spelling is correct) = very unique. great"#request.POST.get('text')

        dataset = 'yelp' # TODO: request.POST.get('dataset')

        # Explain text with configuration above using lime
        context['test_sentence'] = text

        text = text.replace("\n", " ").replace("\r", " ")
        print(type(text))
        print(text)
        global graph
        with graph.as_default():
            if dataset == 'yelp':
                exp = hnatt.explain(h_yelp, text)
            elif dataset=='sentiment':
                exp = hnatt.explain(h_sentiment, text)
            else:
                raise NotImplementedError

        word_attention = exp['word_attention']
        # TODO: colored sentences
        sentence_attention = exp['sentence_attention']
        class_probs = exp['probs']

        classes, words, word_probs = parse_word_attention(word_attention)
        context['wordConfData'] = {'labels': words, 'datasets': build_datasets(classes, word_probs)}

        class_names, cls_proba = parse_hnatt_class_prob(class_probs)
        context["classConfData"] = {'labels': [''], 'datasets': build_datasets(classes, cls_proba)}
        context["out"] = None
    return render(request, 'explainer/hnatt_exp.html', context)