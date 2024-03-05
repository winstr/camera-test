from django.urls import path

from . import views


app_name = 'vision'
urlpatterns = [
    path('', views.vision, name='vision'),
]