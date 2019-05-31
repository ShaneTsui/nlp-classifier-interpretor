from pathlib import Path

from django.shortcuts import render
from django.template import loader


# Create your views here.
from django.http import HttpResponse
from modules.yelp_predict_with_lime import explain_sentence
from modules.html_writer import *

def explainer(request):
    # template = loader.get_template('explainer/index.html')
    context = {
        'latest_question_list': [1, 2, 3],
    }

    if request.method == 'POST':
        your_name = request.POST.get('text')
        your_name = your_name.replace("\n", " ").replace("\r", " ")
        print(your_name)
        exp = explain_sentence(your_name)
        output_filename = Path(__file__).parent / "explanation.html"
        bundle_js, out = save_to_file(exp, output_filename, show_predicted_value=False)
        context = {"out" : out, "bundle_js":bundle_js}


    # return HttpResponse(template.render(context, request))
    return render(request, 'explainer/explanation.html', context)

def index(request):
    return render(request, 'explainer/index.html', )