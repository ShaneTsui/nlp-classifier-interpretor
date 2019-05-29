from django.shortcuts import render
from django.template import loader

# Create your views here.
from django.http import HttpResponse

def index(request):
    # template = loader.get_template('explainer/index.html')
    context = {
        'latest_question_list': [1, 2, 3],
    }
    # return HttpResponse(template.render(context, request))
    return render(request, 'explainer/explanation.html', context)



