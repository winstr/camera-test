from django.shortcuts import render


def vision(request):
    return render(request, 'vision/vision.html')