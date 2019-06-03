from django.urls import path

from . import views

# urlpatterns = [
#     path('', views.explainer, name='explainer'),
#     path('index', views.index, name='index'),
# ]

urlpatterns = [
    # path('', views.explainer, name='explainer'),
    path('lime/', views.lime_explain, name='lime'),
    path('hnatt/', views.hnatt_explain, name='hnatt')
]